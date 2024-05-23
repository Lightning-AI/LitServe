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
import sys
import re
import pytest
import selectors
import time

from tqdm import tqdm

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


def get_extra_time(content: str) -> int:
    if "torch" in content or "transformers" in content:
        return 5

    return 0


def run_script_with_timeout(file, timeout, extra_time, killall):
    sel = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            ["python", str(file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # Line-buffered
            universal_newlines=True,  # Decode bytes to string
        )

        stdout_lines = []
        stderr_lines = []
        end_time = time.time() + timeout + extra_time

        sel.register(process.stdout, selectors.EVENT_READ)
        sel.register(process.stderr, selectors.EVENT_READ)

        while True:
            timeout_remaining = end_time - time.time()
            if timeout_remaining <= 0:
                killall(process)
                break

            events = sel.select(timeout=timeout_remaining)
            for key, _ in events:
                if key.fileobj is process.stdout:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                elif key.fileobj is process.stderr:
                    line = process.stderr.readline()
                    if line:
                        stderr_lines.append(line)

            if process.poll() is not None:
                break

        output = "".join(stdout_lines)
        errors = "".join(stderr_lines)

        # Get the return code of the process
        returncode = process.returncode

    except Exception as e:
        output = ""
        errors = str(e)
        returncode = -1  # Indicate failure in running the process

    return returncode, output, errors


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows CI is slow and this test is just a sanity check.")
def test_readme(tmp_path, killall):
    d = tmp_path / "readme_codes"
    d.mkdir(exist_ok=True)
    code_blocks = get_code_blocks("README.md")
    assert len(code_blocks) > 0, "No code block found in README.md"

    for i, code in enumerate(tqdm(code_blocks)):
        file = d / f"{i}.py"
        file.write_text(code)
        extra_time = get_extra_time(code)

        returncode, stdout, stderr = run_script_with_timeout(file, timeout=5, extra_time=extra_time, killall=killall)

        if "server.run" in code:
            assert uvicorn_msg in stderr, (
                f"Expected to run uvicorn server.\n" f"Code:\n {code}\n\nCode output: {stderr}"
            )
        elif "requests.post" in code:
            assert "ConnectionError" in stderr, (
                f"Client examples should fail with a ConnectionError because there is no server running."
                f"\nCode:\n{code}"
            )
        else:
            assert returncode == 0, (
                f"Code exited with {returncode}.\n"
                f"Error: {stderr}\n"
                f"Please check the code for correctness:\n```\n{code}\n```"
            )
