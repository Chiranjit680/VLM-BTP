"""Thin wrapper around Qwen2.5-VL-3B loaded from the local models folder."""
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from . import config, logs

log = logs.get("model")

_singleton = None


def _check_cudnn():
    """Disable cuDNN if it cannot initialise.

    This env has nvidia-cudnn-cu13 installed over nvidia-cudnn-cu12; both ship
    libcudnn.so.9, so torch (cu121) loads the CUDA 13 build and every cuDNN call
    fails with CUDNN_STATUS_NOT_INITIALIZED. Qwen2.5-VL only uses cuDNN for the
    vision patch-embed conv3d, so falling back costs almost nothing. The probe
    leaves a healthy environment untouched.
    """
    if not (torch.cuda.is_available() and torch.backends.cudnn.enabled):
        return
    log.debug("probing cuDNN")
    try:
        conv = torch.nn.Conv3d(3, 8, 1, bias=False).cuda().half()
        conv(torch.zeros(1, 3, 1, 2, 2, device="cuda", dtype=torch.float16))
    except RuntimeError as e:
        torch.backends.cudnn.enabled = False
        log.warning("cuDNN unusable (%s); disabled it, conv3d falls back to a native kernel",
                    str(e).splitlines()[0])


class QwenVL:
    def __init__(self, model_dir=config.MODEL_DIR):
        _check_cudnn()
        # bf16 only on Ampere+ (A100 = sm_80). torch.cuda.is_bf16_supported() also
        # returns True on the V100s here, where bf16 is emulated and slow.
        native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if native_bf16 else torch.float16
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        with logs.timed(log, f"loading Qwen2.5-VL on {device} as {str(dtype).split('.')[-1]}"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype=dtype, device_map="auto"
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                model_dir, min_pixels=config.MIN_PIXELS, max_pixels=config.MAX_PIXELS
            )
        if torch.cuda.is_available():
            log.info("GPU memory held after load: %.1f GiB", torch.cuda.memory_allocated() / 2**30)

    @torch.inference_mode()
    def chat(self, prompt: str, image: Image.Image | None = None,
             system: str | None = None, max_new_tokens=config.MAX_NEW_TOKENS) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        content = [{"type": "image"}] if image is not None else []
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image] if image is not None else None,
            return_tensors="pt",
        ).to(self.model.device)
        n_in = inputs.input_ids.shape[1]
        log.debug("generate: %d prompt tokens, image=%s, max_new=%d",
                  n_in, image.size if image is not None else None, max_new_tokens)
        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, n_in:]
        dt = time.perf_counter() - t0
        n_out = out.shape[1]
        log.debug("generate: %d tokens in %.1fs (%.1f tok/s)", n_out, dt, n_out / dt if dt else 0)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def get_model() -> QwenVL:
    global _singleton
    if _singleton is None:
        _singleton = QwenVL()
    return _singleton
