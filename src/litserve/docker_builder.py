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
import logging
import os
import sys
import warnings
import litserve as ls

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# COLOR CODES
BOLD = "\u001b[1m"
RESET = "\u001b[0m"
RED = "\u001b[31m"
GREEN = "\u001b[32m"
BLUE = "\u001b[34m"
MAGENTA = "\u001b[35m"

# ACTION CODES
INFO = f"{BOLD}{BLUE}[INFO]{RESET}"
BOLD_ACT = f"{BOLD}{MAGENTA}"
WARNING = f"{BOLD}{RED}[WARNING]{RESET}"


def color(text, color_code, action_code=None):
    if action_code:
        return f"{action_code} {color_code}{text}{RESET}"
    return f"{color_code}{text}{RESET}"


REQUIREMENTS_FILE = "requirements.txt"
DOCKERFILE_TEMPLATE = """
FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir litserve=={version} {requirements}
EXPOSE {port}
CMD ["python", "/app/{server_path}"]
"""

SUCCESS_MSG = """
Dockerfile created successfully at {dockerfile_path}.
- To build the Docker image, run: docker build -t <tag> .
- To run the Docker container, run: docker run -p <host_port>:<container_port> <tag>
- To push the Docker image to a registry, run: docker push <tag>
"""


def build(server_path: str, port: int = 8000):
    """Build a Docker image from the given server code.

    Args:
        server_path (str): The path to the server file.
        port (int, optional): The port to expose in the Docker container. Defaults

    """
    files = []
    requirements = ""
    if os.path.exists(REQUIREMENTS_FILE):
        requirements = f"-r {REQUIREMENTS_FILE}"
        files.append(REQUIREMENTS_FILE)
    else:
        warnings.warn(
            f"requirements.txt not found at {os.getcwd()}. "
            f"Make sure to install the required packages in the Dockerfile."
        )
    files.append(server_path)  # TODO: Make it flexible
    for file in files:
        if not os.path.exists(file):
            logger.error(f"File not found: {file}")
            sys.exit(1)

    version = ls.__version__
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        server_path=server_path, port=port, version=version, requirements=requirements
    )
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    success_msg = SUCCESS_MSG.format(dockerfile_path=os.path.abspath("Dockerfile"))
    print(success_msg)
