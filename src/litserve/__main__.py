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
from jsonargparse import set_config_read_mode, set_docstring_parse_options, CLI

from litserve.docker_build import build


def main():
    cli_components = {"build": build}
    set_docstring_parse_options(attribute_docstrings=True)
    set_config_read_mode(urls_enabled=True)
    CLI(cli_components)


if __name__ == "__main__":
    main()
