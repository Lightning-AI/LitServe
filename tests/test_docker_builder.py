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
from litserve import docker_builder


def test_color():
    assert docker_builder.color("hi", docker_builder.RED) == f"{docker_builder.RED}hi{docker_builder.RESET}"

    expected = f"{docker_builder.INFO} {docker_builder.RED}hi{docker_builder.RESET}"
    assert docker_builder.color("hi", docker_builder.RED, docker_builder.INFO) == expected


def test_build(tmp_path):
    with open(tmp_path / "app.py", "w") as f:
        f.write("print('hello')")
    with open(tmp_path / "requirements.txt", "w") as f:
        f.write("litserve")

    docker_builder.build(tmp_path / "app.py", 8000)
    with open("Dockerfile") as f:
        content = f.read()
    assert (
        """FROM python:3.10-slim
WORKDIR /app
COPY . /app"""
        in content
    )
    assert "EXPOSE 8000" in content
