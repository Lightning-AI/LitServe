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


EXPECTED_CONENT = f"""FROM python:3.10-slim

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


def test_build(tmp_path, monkeypatch):
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

    with pytest.raises(FileNotFoundError, match="must be in the current directory"):
        docker_builder.dockerize("random_file_name.py", 8000)
