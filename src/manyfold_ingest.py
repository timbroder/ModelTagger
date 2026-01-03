import csv
import requests
import os


def model_exists(api_url: str, headers: dict, model_name: str) -> bool:
    """Check if a model with the given name already exists in Manyfold."""
    response = requests.get(
        f"{api_url}/models?name={model_name}",
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        return any(model.get("name") == model_name for model in data)
    return False


def run_upload(csv_path: str) -> None:
    """Upload tags from CSV to Manyfold.

    NOTE: This function currently only logs intended uploads.
    Actual file upload and tag posting to Manyfold API is not yet implemented.
    To complete this, add requests to POST model files and PATCH tags.
    """
    api_url = os.getenv("MANYFOLD_API_URL")
    token = os.getenv("MANYFOLD_API_TOKEN")

    if not api_url or not token:
        print("Error: MANYFOLD_API_URL and MANYFOLD_API_TOKEN environment variables required.")
        return

    headers = {"Authorization": f"Bearer {token}"}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["filename"]
            tags = row["tags"]
            if model_exists(api_url, headers, name):
                print(f"Skipping {name}, already exists.")
                continue
            # TODO: Implement actual upload - POST model file, then PATCH to add tags
            print(f"[DRY RUN] Would upload {name} with tags: {tags}")
