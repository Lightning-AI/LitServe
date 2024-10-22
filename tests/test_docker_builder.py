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
import pytest

import litserve as ls
from litserve import docker_builder


def test_color():
    assert docker_builder.color("hi", docker_builder.RED) == f"{docker_builder.RED}hi{docker_builder.RESET}"

    expected = f"{docker_builder.INFO} {docker_builder.RED}hi{docker_builder.RESET}"
    assert docker_builder.color("hi", docker_builder.RED, docker_builder.INFO) == expected


EXPECTED_CONENT = f"""ARG PYTHON_VERSION=3.12
FROM python:$PYTHON_VERSION-slim

####### Add your own installation commands here #######
# RUN pip install some-package
# RUN wget https://path/to/some/data/or/weights
# RUN apt-get update && apt-get install -y <package-name>

WORKDIR /app
COPY . /app

# Install litserve and requirements
RUN pip install --no-cache-dir litserve=={ls.__version__} -r requirements.txt
EXPOSE 8000
CMD ["python", "/app/app.py"]
"""


EXPECTED_GPU_DOCKERFILE = f"""# Change CUDA and cuDNN version here
ARG PYTHON_VERSION=3.12
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

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
RUN pip install --no-cache-dir litserve=={ls.__version__} -r requirements.txt
EXPOSE 8000
CMD ["python", "/app/app.py"]
"""


def test_dockerize(tmp_path, monkeypatch):
    with open(tmp_path / "app.py", "w") as f:
        f.write("print('hello')")

    # Temporarily change the current working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    with pytest.warns(UserWarning, match="Make sure to install the required packages in the Dockerfile."):
        docker_builder.dockerize("app.py", 8000)

    with open(tmp_path / "requirements.txt", "w") as f:
        f.write("lightning")

    docker_builder.dockerize("app.py", 8000)
    with open("Dockerfile") as f:
        content = f.read()
    assert content == EXPECTED_CONENT

    docker_builder.dockerize("app.py", 8000, gpu=True)
    with open("Dockerfile") as f:
        content = f.read()
    assert content == EXPECTED_GPU_DOCKERFILE

    with pytest.raises(FileNotFoundError, match="must be in the current directory"):
        docker_builder.dockerize("random_file_name.py", 8000)
