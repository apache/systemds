import argparse
import math
import os
import requests


def list_workflow_artifacts(
    owner_repo: str,
    run_id: str,
    token: str,
    per_page: int = 30,
    page: int = 1,
) -> requests.Response:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url_base = "https://api.github.com"
    query_params = f"per_page={per_page}&page={page}"
    url = (
        f"{url_base}/repos/{owner_repo}/actions/runs/{run_id}/artifacts?{query_params}"
    )

    return requests.get(url=url, headers=headers)


def delete_artifact(
    owner_repo: str,
    artifact_id: str,
    token: str,
) -> requests.Response:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    url_base = "https://api.github.com"
    url = f"{url_base}/repos/{owner_repo}/actions/artifacts/{artifact_id}"

    return requests.delete(url=url, headers=headers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deletes Artifacts")
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="From github action pass ${{ secrets.GITHUB_TOKEN }}",
    )
    parser.add_argument(
        "-o",
        "--owner-repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", None),
        help="Defaults to envvar 'GITHUB_REPOSITORY'",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        default=os.getenv("GITHUB_RUN_ID"),
        help="Defaults to envvar 'GITHUB_RUN_ID'",
    )

    args = parser.parse_args()

    page = 1
    items_per_page = 85
    resp = list_workflow_artifacts(
        owner_repo=args.owner_repo,
        run_id=args.run_id,
        token=args.token,
        page=page,
        per_page=items_per_page,
    )

    resp_dict = resp.json()
    artifacts_count = resp_dict.get("total_count")
    artifact_ids = [x.get("id") for x in resp_dict.get("artifacts")]

    if items_per_page < artifacts_count:
        pages = math.ceil(artifacts_count / items_per_page)
        for page in range(2, pages + 1):
            resp = list_workflow_artifacts(
                owner_repo=args.owner_repo,
                run_id=args.run_id,
                token=args.token,
                page=page,
                per_page=items_per_page,
            )
            [artifact_ids.append(x.get("id")) for x in resp.json().get("artifacts")]

    for artifact_id in artifact_ids:
        delete_artifact(
            owner_repo=args.owner_repo,
            artifact_id=artifact_id,
            token=args.token,
        )
