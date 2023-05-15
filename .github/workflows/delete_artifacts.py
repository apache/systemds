#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

import argparse
import logging
import math
import os
import requests

logging.basicConfig(level=logging.INFO)

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

    logging.info(f"Deleting artifact at url: {url}")
    return requests.delete(url=url, headers=headers)


if __name__ == "__main__":
    logging.info("Running delete-artifacts.py")
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
    if args.token is None:
        logging.info(f"--token: is not set! Aborting")
        exit(1)
    else:
        logging.info(f"--token: is set, continue")
    logging.info(f"--owner-repository: {args.owner_repo}")
    logging.info(f"--run-id: {args.run_id}")


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
    logging.info(f"Artifacts count: {len(artifact_ids)} of {artifacts_count}")

    if items_per_page < artifacts_count:
        pages = math.ceil(artifacts_count / items_per_page)
        logging.info(f"Pagecount to retrieve: {pages}")
        for page in range(2, pages + 1):
            resp = list_workflow_artifacts(
                owner_repo=args.owner_repo,
                run_id=args.run_id,
                token=args.token,
                page=page, per_page=items_per_page,
            )
            [artifact_ids.append(x.get("id")) for x in resp.json().get("artifacts")]

    for artifact_id in artifact_ids:
        delete_artifact(
            owner_repo=args.owner_repo,
            artifact_id=artifact_id,
            token=args.token,
        )
