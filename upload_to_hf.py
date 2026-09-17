"""
upload_to_hf.py
Deploys Dataset, Model, and Gradio Space repositories to Hugging Face Hub under khalidml65.
"""
from huggingface_hub import HfApi

api = HfApi()
username = api.whoami()["name"]
print(f"Authenticated as: {username}")

# 1. Upload Dataset Repository
print("\n[1/3] Uploading Dataset Repository...")
dataset_repo = f"{username}/lottery-draw-machine-telemetry"
api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
api.upload_folder(folder_path="data/processed", repo_id=dataset_repo, repo_type="dataset")
api.upload_file(path_or_fileobj="dataset_card.md", path_in_repo="README.md", repo_id=dataset_repo, repo_type="dataset")
print(f"SUCCESS: Dataset at https://huggingface.co/datasets/{dataset_repo}")

# 2. Upload Model Repository
print("\n[2/3] Uploading Model Repository...")
model_repo = f"{username}/lottery-isolation-forest-detector"
api.create_repo(repo_id=model_repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path="models", repo_id=model_repo, repo_type="model")
api.upload_file(path_or_fileobj="model_card.md", path_in_repo="README.md", repo_id=model_repo, repo_type="model")
print(f"SUCCESS: Model at https://huggingface.co/{model_repo}")

# 3. Deploy Gradio Space
print("\n[3/3] Deploying Gradio Space...")
space_repo = f"{username}/lottery-machine-failure-risk"
api.create_repo(repo_id=space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)

for file in ["app.py", "requirements.txt", "README.md", "model_card.md", "dataset_card.md"]:
    api.upload_file(path_or_fileobj=file, path_in_repo=file, repo_id=space_repo, repo_type="space")

for folder in ["src", "models", "configs", "data/raw", "data/processed", "docs"]:
    api.upload_folder(folder_path=folder, path_in_repo=folder, repo_id=space_repo, repo_type="space")

print(f"\nALL DONE! Space is live at: https://huggingface.co/spaces/{space_repo}")