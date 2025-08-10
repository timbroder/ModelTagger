import csv
import requests
import os

def model_exists(api_url, headers, model_name):
    response = requests.get(
        f"{api_url}/models?name={model_name}",
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        return any(model.get("name") == model_name for model in data)
    return False

def run_upload(csv_path):
    api_url = os.getenv("MANYFOLD_API_URL")
    token = os.getenv("MANYFOLD_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["filename"]
            tags = row["tags"]
            if model_exists(api_url, headers, name):
                print(f"Skipping {name}, already uploaded.")
                continue
            # Fake upload logic here — you'd include file upload, tag posting, etc.
            print(f"Uploading {name} with tags: {tags}")
