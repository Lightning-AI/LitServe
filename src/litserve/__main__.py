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
from argparse import ArgumentParser
from litserve.docker_builder import dockerize


def main():
    parser = ArgumentParser(description="CLI for LitServe")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # dockerize sub-command
    dockerize_parser = subparsers.add_parser(
        "dockerize",
        help=dockerize.__doc__,
        description="Generate a Dockerfile for the given server code.",
    )
    dockerize_parser.add_argument(
        "server_filename",
        type=str,
        help="The path to the server file. Example: server.py or app.py.",
    )
    dockerize_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="The port to expose in the Docker container. Defaults to 8000.",
    )
    dockerize_parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Whether to use a GPU-enabled Docker image. Defaults to false.",
    )
    dockerize_parser.set_defaults(func=lambda args: dockerize(args.server_filename, args.port, args.gpu))
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
