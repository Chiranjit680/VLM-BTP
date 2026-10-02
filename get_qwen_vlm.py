from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_DIR = "models/qwen2.5-vl-3b-instruct"

path = snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)

print("Path to model files:", path)
