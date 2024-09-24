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
import tarfile
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)

# Your Dockerfile as a string
dockerfile_content = """
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
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


def create_tarball(server_path: str):
    # Create a tar archive in memory
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
        # Add the Dockerfile
        dockerfile_info = tarfile.TarInfo(name="Dockerfile")
        dockerfile_info.size = len(dockerfile_content)
        tar.addfile(dockerfile_info, io.BytesIO(dockerfile_content.encode("utf-8")))

        # Add other files (for example, requirements.txt and app.py)
        with open("requirements.txt", "rb") as req_file:
            tar_info = tarfile.TarInfo(name="requirements.txt")
            tar_info.size = len(req_file.read())
            req_file.seek(0)  # Reset file pointer
            tar.addfile(tar_info, req_file)

        with open(server_path, "rb") as app_file:
            tar_info = tarfile.TarInfo(name="app.py")
            tar_info.size = len(app_file.read())
            app_file.seek(0)  # Reset file pointer
            tar.addfile(tar_info, app_file)

    # Seek to the beginning of the BytesIO buffer
    tar_buffer.seek(0)
    return tar_buffer


def build_docker_image(
    server_path: str, image_name: Optional[str] = None, image_tag: Optional[str] = None, timeout=600
):
    """Build a Docker image from the given server path.

    Args:
        server_path (str): The path to the server file.
        image_name (str): The name of the image.
        image_tag (str): The tag of the image.

    """

    if not image_name:
        image_name = "litserve"
    if not image_tag:
        image_tag = "latest"

    client = get_docker_client()
    try:
        tar_buffer = create_tarball(server_path)
        image, build_logs = client.images.build(fileobj=tar_buffer, tag=f"{image_name}:{image_tag}", timeout=timeout)
        for log in build_logs:
            print(log.get("stream", ""), end="")
        print(f"Image built successfully: {image.id}")
    finally:
        client.close()
