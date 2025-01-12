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
import warnings
from pathlib import Path

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
DOCKERFILE_TEMPLATE = """ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim

####### Add your own installation commands here #######
# RUN pip install some-package
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
COPY . /app

# Install litserve and requirements
RUN pip install --no-cache-dir litserve=={version} {requirements}
EXPOSE {port}
CMD ["python", "/app/{server_filename}"]
"""

CUDA_DOCKER_TEMPLATE = """# Change CUDA and cuDNN version here
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
        software-properties-common \\
        wget \\
    && add-apt-repository ppa:deadsnakes/ppa \\
    && apt-get update && apt-get install -y --no-install-recommends \\
        python$PYTHON_VERSION \\
        python$PYTHON_VERSION-dev \\
        python$PYTHON_VERSION-venv \\
    && wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py \\
    && python$PYTHON_VERSION get-pip.py \\
    && rm get-pip.py \\
    && ln -sf /usr/bin/python$PYTHON_VERSION /usr/bin/python \\
    && ln -sf /usr/local/bin/pip$PYTHON_VERSION /usr/local/bin/pip \\
    && python --version \\
    && pip --version \\
    && apt-get purge -y --auto-remove software-properties-common \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

####### Add your own installation commands here #######
# RUN pip install some-package
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
COPY . /app

# Install litserve and requirements
RUN pip install --no-cache-dir litserve=={version} {requirements}
EXPOSE {port}
CMD ["python", "/app/{server_filename}"]
"""

# Link our documentation as the bottom of this msg
SUCCESS_MSG = """{BOLD}{MAGENTA}Dockerfile created successfully{RESET}
Update {UNDERLINE}{dockerfile_path}{RESET} to add any additional dependencies or commands.{RESET}

{BOLD}Build the container with:{RESET}
> {UNDERLINE}docker build -t litserve-model .{RESET}

{BOLD}To run the Docker container on the machine:{RESET}
> {UNDERLINE}{RUN_CMD}{RESET}

{BOLD}To push the container to a registry:{RESET}
> {UNDERLINE}docker push litserve-model{RESET}
"""


def dockerize(server_filename: str, port: int = 8000, gpu: bool = False):
    """Generate a Dockerfile for the given server code.

    Example usage:
        litserve dockerize server.py --port 8000 --gpu

    Args:
        server_filename (str): The path to the server file. Example sever.py or app.py.
        port (int, optional): The port to expose in the Docker container.
        gpu (bool, optional): Whether to use a GPU-enabled Docker image.

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
    if gpu:
        run_cmd = f"docker run --gpus all -p {port}:{port} litserve-model:latest"
        docker_template = CUDA_DOCKER_TEMPLATE
    else:
        run_cmd = f"docker run -p {port}:{port} litserve-model:latest"
        docker_template = DOCKERFILE_TEMPLATE
    dockerfile_content = docker_template.format(
        server_filename=server_filename,
        port=port,
        version=version,
        requirements=requirements,
    )
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    success_msg = SUCCESS_MSG.format(
        dockerfile_path=os.path.abspath("Dockerfile"),
        RUN_CMD=run_cmd,
        BOLD=BOLD,
        MAGENTA=MAGENTA,
        GREEN=GREEN,
        BLUE=BLUE,
        UNDERLINE=UNDERLINE,
        BG_MAGENTA=BG_MAGENTA,
        RESET=RESET,
    )
    print(success_msg)
