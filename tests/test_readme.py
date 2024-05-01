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
# Code extraction adapted from https://github.com/tassaron/get_code_from_markdown

import subprocess
from typing import List
import re

uvicorn_msg = "Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)"


def extract_code_blocks(lines: List[str]) -> List[str]:
    language = "python"
    regex = re.compile(
        r"(?P<start>^```(?P<block_language>(\w|-)+)\n)(?P<code>.*?\n)(?P<end>```)",
        re.DOTALL | re.MULTILINE,
    )
    blocks = [(match.group("block_language"), match.group("code")) for match in regex.finditer("".join(lines))]
    return [block for block_language, block in blocks if block_language == language]


def get_code_blocks(file: str) -> List[str]:
    with open(file) as f:
        lines = list(f)
        return extract_code_blocks(lines)


def test_readme(tmp_path):
    d = tmp_path / "readme_codes"
    d.mkdir()
    code_blocks = get_code_blocks("README.md")
    assert len(code_blocks) > 0

    for i, code in enumerate(code_blocks):
        file = d / f"{i}.py"
        file.write_text(code)

        try:
            process = subprocess.run(["python", str(file)], capture_output=True, timeout=5)
            errs = process.stderr or b""
        except subprocess.TimeoutExpired as e:
            errs = e.stderr or b""

        errs = errs.decode("utf-8")
        if "server.run" in code:
            assert uvicorn_msg in errs, f"Expected to run uvicorn server.\n Outputs: {errs}"
        elif "requests.post" in code:
            assert (
                "requests.exceptions.ConnectionError" in errs
            ), "Client examples should fail with a ConnectionError because there is no server running."

        else:
            assert process.returncode == 0, (
                f"Code exited with {process.returncode}.\n Please check the code for correctness:\n" f"```\n{code}\n```"
            )
