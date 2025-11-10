import argparse
import zipfile
from pathlib import Path

import requests


def download_artifacts(token, owner, repo, run_id, output_dir):
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    # Get list of artifacts
    artifacts_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    response = requests.get(artifacts_url, headers=headers)
    response.raise_for_status()
    artifacts = response.json()["artifacts"]

    if not artifacts:
        print(f"No artifacts found for run ID: {run_id}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(artifacts)} artifacts")
    for artifact in artifacts:
        name = artifact["name"]
        download_url = artifact["archive_download_url"]

        print(f"Downloading {name}...")

        response = requests.get(download_url, headers=headers, stream=True)
        response.raise_for_status()

        temp_zip = output_dir / f"{name}.zip"
        with open(temp_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        temp_zip.unlink()
        print(f"Downloaded and extracted {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download artifacts from a GitHub Actions run")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--owner", required=True, help="Repository owner")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--run-id", required=True, help="Workflow run ID")
    parser.add_argument(
        "--output-dir", default="artifacts", help="Output directory for downloaded artifacts"
    )

    args = parser.parse_args()

    download_artifacts(args.token, args.owner, args.repo, args.run_id, args.output_dir)
