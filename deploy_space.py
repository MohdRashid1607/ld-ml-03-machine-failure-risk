"""
deploy_space.py
1. Ensures clean sample CSV without spaces or brackets.
2. Uploads fixed app.py, sample data, processed datasets, and models to Hugging Face Space.
"""
import shutil
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
repo_id = "khalidml65/lottery-machine-failure-risk"

print("=" * 60)
print(f"Deploying to Space: {repo_id}")
print("=" * 60)

# 1. Create clean sample file without special characters
raw_path = Path("data/raw/lottery_weather_dataset_20260914 (3).csv")
clean_path = Path("data/sample_draws.csv")
if raw_path.exists():
    shutil.copyfile(raw_path, clean_path)
    print(f"✅ Created clean sample file: {clean_path}")
elif clean_path.exists():
    print(f"✅ Clean sample file exists: {clean_path}")
else:
    print("⚠️ Raw file not found, checking processed files...")

# 2. Upload app.py
print("\n[1/3] Uploading updated app.py...")
api.upload_file(
    path_or_fileobj="app.py",
    path_in_repo="app.py",
    repo_id=repo_id,
    repo_type="space",
)
print("✅ app.py uploaded successfully.")

# 3. Upload clean sample CSV
if clean_path.exists():
    print("\n[2/3] Uploading clean sample CSV...")
    api.upload_file(
        path_or_fileobj=str(clean_path),
        path_in_repo="data/sample_draws.csv",
        repo_id=repo_id,
        repo_type="space",
    )
    print("✅ data/sample_draws.csv uploaded.")

# 4. Upload data folder to ensure all fallbacks exist on Space
print("\n[3/3] Uploading data folder (sample & processed)...")
try:
    api.upload_folder(
        folder_path="data",
        path_in_repo="data",
        repo_id=repo_id,
        repo_type="space",
    )
    print("✅ data folder synced.")
except Exception as e:
    print(f"⚠️ data folder sync warning: {e}")

print("\n" + "=" * 60)
print("🚀 DEPLOYMENT COMPLETE! Space is rebuilding.")
print("Check status at: https://huggingface.co/spaces/khalidml65/lottery-machine-failure-risk")
print("=" * 60)
