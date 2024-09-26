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
from pathlib import Path

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
RESET = "\u001b[0m"
RED = "\u001b[31m"
GREEN = "\u001b[32m"
BLUE = "\u001b[34m"
MAGENTA = "\u001b[35m"
BG_MAGENTA = "\u001b[45m"

# ACTION CODES
BOLD = "\u001b[1m"
UNDERLINE = "\u001b[4m"
INFO = f"{BOLD}{BLUE}[INFO]"
WARNING = f"{BOLD}{RED}[WARNING]"


def color(text, color_code, action_code=None):
    if action_code:
        return f"{action_code} {color_code}{text}{RESET}"
    return f"{color_code}{text}{RESET}"


REQUIREMENTS_FILE = "requirements.txt"
DOCKERFILE_TEMPLATE = """FROM python:3.10-slim

####### Put installation commands here #######
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
COPY . /app

# Install litserve and requirements
RUN pip install --no-cache-dir litserve=={version} {requirements}
EXPOSE {port}
CMD ["python", "/app/{server_filename}"]
"""

# Link our documentation as the bottom of this msg
SUCCESS_MSG = """
{BOLD}{MAGENTA}Dockerfile created successfully at{RESET} {UNDERLINE}{dockerfile_path}{RESET}

{BOLD}{BLUE}Follow the instructions below to build & run a docker image:{RESET}
- To build the Docker image, run: {UNDERLINE}docker build -t <tag> .{RESET}
- To run the Docker container, run: {UNDERLINE}docker run -p <host_port>:<container_port> <tag>{RESET}
- To push the Docker image to a registry, run: {UNDERLINE}docker push <tag>{RESET}
"""


def build(server_filename: str, port: int = 8000):
    """Build a Docker image from the given server code.

    Args:
        server_filename (str): The path to the server file. Example sever.py or app.py.
        port (int, optional): The port to expose in the Docker container. Defaults to 8000.

    """
    requirements = ""
    if os.path.exists(REQUIREMENTS_FILE):
        requirements = f"-r {REQUIREMENTS_FILE}"
    else:
        warnings.warn(
            f"requirements.txt not found at {os.getcwd()}. "
            f"Make sure to install the required packages in the Dockerfile.",
            UserWarning,
        )

    current_dir = Path.cwd()
    if not (current_dir / server_filename).is_file():
        raise FileNotFoundError(f"Server file `{server_filename}` must be in the current directory: {os.getcwd()}")

    version = ls.__version__
    dockerfile_content = DOCKERFILE_TEMPLATE.format(
        server_filename=server_filename, port=port, version=version, requirements=requirements
    )
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    success_msg = SUCCESS_MSG.format(
        dockerfile_path=os.path.abspath("Dockerfile"),
        port=port,
        BOLD=BOLD,
        MAGENTA=MAGENTA,
        GREEN=GREEN,
        BLUE=BLUE,
        UNDERLINE=UNDERLINE,
        BG_MAGENTA=BG_MAGENTA,
        RESET=RESET,
    )
    print(success_msg)
