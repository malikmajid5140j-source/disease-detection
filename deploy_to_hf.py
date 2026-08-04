import os
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
api = HfApi(token=token)

try:
    user_info = api.whoami()
    username = user_info['name']
    print(f"Authenticated as: {username}")
except Exception as e:
    print("Authentication failed. Please check your token.")
    exit(1)

repo_name = "agriscan-ai"
repo_id = f"{username}/{repo_name}"

print(f"Creating/Checking Space: {repo_id}...")
try:
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True, token=token)
    print("Space ready!")
except Exception as e:
    print(f"Error creating space: {e}")
    exit(1)

print("Uploading files... This may take a few minutes depending on your model size.")
try:
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=[".git*", "__pycache__", ".venv", "deploy_to_hf.py"]
    )
    print("--------------------------------------------------")
    print(f"DEPLOYMENT SUCCESSFUL!")
    print(f"Your app is now live at: https://huggingface.co/spaces/{repo_id}")
    print("--------------------------------------------------")
except Exception as e:
    print(f"Failed to upload files: {e}")
