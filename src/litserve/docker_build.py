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
import io
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import List, Optional

import docker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

# Add a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DOCKERFILE_CONTENT = """FROM python:3.9-slim
WORKDIR /app
COPY . /app

RUN pip install litserve -r requirements.txt
EXPOSE {port}
CMD ["python", "/app/{server_path}"]
"""


def get_docker_client() -> docker.APIClient:
    """Initializes and returns a low-level Docker API client. If the Docker SDK is not installed, prompts the user to
    install it.

    Returns:
        docker.APIClient: An instance of Docker's low-level API client.

    """
    try:
        import docker

        client = docker.APIClient(base_url="unix://var/run/docker.sock")
        client.ping()
        logger.debug("Successfully connected to Docker daemon.")
        return client
    except ImportError:
        response = input("Docker client is not installed. Do you want to install it now? (yes/no): ").strip().lower()
        if response in ["yes", "y"]:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "docker"])
            logger.info("Docker client installed successfully. Please rerun the script.")
            sys.exit(0)
        else:
            logger.error("Docker client installation skipped. Exiting.")
            sys.exit(1)
    except docker.errors.APIError as e:
        logger.error(f"Failed to connect to Docker daemon: {e}")
        sys.exit(1)


def build_docker_image_with_tempdir(
    client: docker.APIClient, dockerfile_str: str, files: List[str], tag: str, timeout: int
) -> Optional[str]:
    """Builds a Docker image using a temporary directory as the build context.

    Args:
        client (docker.APIClient): The low-level Docker API client.
        dockerfile_str (str): The content of the Dockerfile.
        files (List[str]): List of file paths to include in the build context.
        tag (str): The tag for the resulting Docker image.
        timeout (int): The build timeout in seconds.

    Returns:
        Optional[str]: The image ID if the build is successful, None otherwise.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write Dockerfile
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        try:
            with open(dockerfile_path, "w", encoding="utf-8") as df:
                df.write(dockerfile_str)
            logger.debug("Dockerfile written successfully.")
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
            raise

        # Create a tar archive of the temporary directory
        try:
            tar_stream = create_tar_stream(tmpdir)
        except Exception as e:
            logger.error(f"Failed to create tar archive: {e}")
            return None

        # Build the Docker image using the low-level API
        try:
            logger.info(f"Starting build for image '{tag}'...")
            build_logs = client.build(
                fileobj=tar_stream, custom_context=True, tag=tag, rm=True, decode=True, timeout=timeout
            )
            image_id = None
            for chunk in build_logs:
                if "stream" in chunk:
                    log_message = chunk["stream"].strip()
                    if log_message:
                        logger.info(log_message)
                elif "error" in chunk:
                    logger.error(chunk["error"].strip())
            logger.info(f"Image '{tag}' built successfully.")
            # Optionally, retrieve the image ID
            images = client.images(name=tag)
            if images:
                image_id = images[0]["Id"]
                logger.debug(f"Image ID: {image_id}")
            return image_id
        except docker.errors.BuildError as e:
            logger.error(f"Build failed: {e}")
            return None
        except docker.errors.APIError as e:
            logger.error(f"Docker API error during build: {e}")
            return None
        finally:
            tar_stream.close()


def create_tar_stream(directory: str) -> io.BytesIO:
    """Creates a gzip-compressed tar archive from the specified directory.

    Args:
        directory (str): The path to the directory to archive.

    Returns:
        io.BytesIO: An in-memory tar archive.

    """
    import io

    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
        tar.add(directory, arcname=".")
    tar_stream.seek(0)
    return tar_stream


def build(server_path: str, tag: Optional[str] = None, port: int = 8000, timeout: int = 600):
    """Build a Docker image from the given server code.

    Args:
        server_path (str): The path to the server file.
        tag (str, optional): The tag of the Docker image. Defaults to 'litserve:latest'.
        port (int, optional): The port to expose. Defaults to 8000.
        timeout (int, optional): The timeout for building the Docker image in seconds. Defaults to 600.

    """
    if not tag:
        tag = "litserve:latest"

    client = get_docker_client()

    files = [server_path, "requirements.txt"]  # TODO: Make it flexible
    for file in files:
        if not os.path.exists(file):
            logger.error(f"File not found: {file}")
            sys.exit(1)

    try:
        dockerfile_content = DOCKERFILE_CONTENT.format(server_path=server_path, port=port)
        image_id = build_docker_image_with_tempdir(
            client=client, dockerfile_str=dockerfile_content, files=files, tag=tag, timeout=timeout
        )
        if image_id:
            print(f"Image '{tag}' built successfully with ID: {image_id}")
        else:
            logger.error("Docker image build did not complete successfully.")
    finally:
        client.close()
