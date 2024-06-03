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
import json
import os
import psutil
import requests
import subprocess
import time

from openai import OpenAI


def e2e_from_file(filename):
    def decorator(test_fn):
        def wrapper(*args, **kwargs):
            process = subprocess.Popen(
                ["python", filename],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
            time.sleep(5)
 
            try:
                test_fn(*args, **kwargs)
            except Exception as e:
                raise
            finally:
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                process.kill()
        return wrapper
    return decorator


@e2e_from_file("tests/e2e/default_api.py")
def test_e2e_default_api():
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    assert resp.json() == {"output": 16.0}, "tests/simple_server.py didn't return expected output"


@e2e_from_file("tests/e2e/default_spec.py")
def test_e2e_default_spec(openai_request_data):
    resp = requests.post("http://127.0.0.1:8000/v1/chat/completions", json=openai_request_data)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    output = resp.json()["choices"][0]["message"]["content"]
    expected = "This is a generated output"
    assert output == expected, "tests/default_spec.py didn't return expected output"


@e2e_from_file("tests/e2e/default_batching.py")
def test_e2e_default_batching():
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    assert resp.json() == {"output": 16.0}, "tests/simple_server.py didn't return expected output"


@e2e_from_file("tests/e2e/default_batched_streaming.py")
def test_e2e_batched_streaming():
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None, stream=True)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"

    outputs = []
    for line in resp.iter_content(chunk_size=4000):
        if line:
            outputs.append(json.loads(line.decode("utf-8")))

    assert len(outputs) == 10, "streaming server should have 10 outputs"
    assert {"output": 16.0} in outputs, "server didn't return expected output"


@e2e_from_file("tests/e2e/default_openaispec.py")
def test_openai_parity():
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="lit",  # required, but unused
    )
    response = client.chat.completions.create(
        model="lit",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How are you?"},
        ],
    )
    assert response.choices[0].message.content == "This is a generated output", (
        f"Server didn't return expected output" f"\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How are you?"},
        ],
        stream=True,
    )

    expected_outputs = ["This is a generated output", None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\n" f"OpenAI client output: {r}"
        )


@e2e_from_file("tests/e2e/default_openaispec.py")
def test_openai_parity_with_image_input():
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="lit",  # required, but unused
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            ],
        },
    ]
    response = client.chat.completions.create(
        model="lit",
        messages=messages,
    )
    assert response.choices[0].message.content == "This is a generated output", (
        f"Server didn't return expected output" f"\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        stream=True,
    )

    expected_outputs = ["This is a generated output", None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\n" f"OpenAI client output: {r}"
        )


@e2e_from_file("tests/e2e/default_openaispec_tools.py")
def test_openai_parity_with_tools():
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="lit",  # required, but unused
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "What's the weather like in Boston today?"},
    ]
    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        tools=tools,
    )
    assert response.choices[0].message.content == "", (
        f"Server didn't return expected output" f"\nOpenAI client output: {response}"
    )
    assert response.choices[0].message.tool_calls[0].function.name == "get_current_weather", (
        f"Server didn't return expected output" f"\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        stream=True,
    )

    expected_outputs = ["", None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\n" f"OpenAI client output: {r}"
        )
        if r.choices[0].delta.tool_calls:
            assert r.choices[0].delta.tool_calls[0].function.name == "get_current_weather", (
                f"Server didn't return expected output.\n" f"OpenAI client output: {r}"
            )


@e2e_from_file("tests/simple_server.py")
def test_run():
    assert os.path.exists(
        "client.py"
    ), f"Expected client file to be created at {os.getcwd()} after starting the server"
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output, f"tests/simple_server.py didn't return expected output, got {output}"
    os.remove("client.py")
