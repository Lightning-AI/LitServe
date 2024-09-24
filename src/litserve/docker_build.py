# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import subprocess
import docker
import tempfile
import os
import shutil
import logging
import sys
from typing import Optional, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Your Dockerfile as a string
DOCKERFILE_CONTENT = """
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", /app/{server_path}]
"""


def get_docker_client():
    try:
        import docker

        return docker.from_env()
    except ImportError:
        response = input("Docker client is not installed. Do you want to install it now? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "docker"])
            print("Docker client installed successfully.")
        else:
            print("Docker client installation skipped. Exiting.")
            sys.exit(1)


def build_docker_image_with_tempdir(
    client: docker.DockerClient, dockerfile_str: str, files: List[str], tag: str, timeout: int
):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write Dockerfile
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        try:
            with open(dockerfile_path, "w", encoding="utf-8") as df:
                print(dockerfile_str)
                df.write(dockerfile_str)
            logger.info("Dockerfile written successfully.")
        except Exception as e:
            logger.error(f"Failed to write Dockerfile: {e}")
            return None

        # Copy other files
        try:
            for file_path in files:
                shutil.copy(file_path, tmpdir)
                logger.info(f"Copied {file_path} to {tmpdir}.")
        except Exception as e:
            logger.error(f"Failed to copy files: {e}")
            return None

        # Build the Docker image
        try:
            return client.images.build(path=tmpdir, tag=tag, rm=True, timeout=timeout)
        except docker.errors.BuildError as e:
            logger.error(f"Build failed: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise


def build(server_path: str, tag: Optional[str] = None, timeout=600):
    """Build a Docker image from the given server code.

    Args:
        server_path (str): The path to the server file.
        tag (str): The tag of the docker image. Defaults to litserve:latest.
        timeout (int): The timeout for building the Docker image. Defaults to 600 seconds.

    """

    if not tag:
        tag = "litserve:latest"

    client = get_docker_client()

    files = [server_path, "requirements.txt"]  # TODO: Make it flexible
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"file not found at: {file}")

    try:
        dockerfile_content = DOCKERFILE_CONTENT.format(server_path=server_path)
        image, logs = build_docker_image_with_tempdir(client, dockerfile_content, files, tag, timeout=timeout)
        for log in logs:
            if "stream" in log:
                logger.info(log["stream"].strip())
        print(f"Image {tag} built successfully: {image.id}")
    finally:
        client.close()
