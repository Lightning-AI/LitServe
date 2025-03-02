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
import subprocess
import time
from functools import wraps

import psutil
import requests
from openai import OpenAI


def e2e_from_file(filename):
    def decorator(test_fn):
        @wraps(test_fn)
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
            except Exception:
                raise
            finally:
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                process.kill()

        return wrapper

    return decorator


@e2e_from_file("tests/simple_server.py")
def test_run():
    assert os.path.exists("client.py"), f"Expected client file to be created at {os.getcwd()} after starting the server"
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output, f"tests/simple_server.py didn't return expected output, got {output}"
    os.remove("client.py")


@e2e_from_file("tests/simple_server_diff_port.py")
def test_run_with_port():
    assert os.path.exists("client.py"), f"Expected client file to be created at {os.getcwd()} after starting the server"
    with open(os.path.join(os.getcwd(), "client.py")) as f:
        client_code = f.read()
        assert ":8080" in client_code, "Could not find 8080 in client.py"
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output, (
        f"tests/simple_server_server_diff_port.py didn't return expected output, got {output}"
    )
    os.remove("client.py")


@e2e_from_file("tests/e2e/default_api.py")
def test_e2e_default_api():
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    assert resp.json() == {"output": 16.0}, "tests/simple_server.py didn't return expected output"


@e2e_from_file("tests/e2e/default_multi_worker_api.py")
def test_e2e_default_multi_worker_api():
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"
    assert resp.json() == {"output": 16.0}, "tests/e2e/default_multi_worker_api.py didn't return expected output"


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
        f"Server didn't return expected output\nOpenAI client output: {response}"
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
            f"Server didn't return expected output.\nOpenAI client output: {r}"
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
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        "detail": "low",
                    },
                },
            ],
        },
    ]
    response = client.chat.completions.create(
        model="lit",
        messages=messages,
    )
    assert response.choices[0].message.content == "This is a generated output", (
        f"Server didn't return expected output\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        stream=True,
    )

    expected_outputs = ["This is a generated output", None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\nOpenAI client output: {r}"
        )


@e2e_from_file("tests/e2e/default_openaispec.py")
def test_openai_parity_with_audio_input(openai_request_data_with_audio_wav):
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="lit",  # required, but unused
    )
    messages = openai_request_data_with_audio_wav["messages"]
    response = client.chat.completions.create(
        model="lit",
        messages=messages,
    )
    assert response.choices[0].message.content == "This is a generated output", (
        f"Server didn't return expected output\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        stream=True,
    )

    expected_outputs = ["This is a generated output", None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\nOpenAI client output: {r}"
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
        f"Server didn't return expected output\nOpenAI client output: {response}"
    )
    assert response.choices[0].message.tool_calls[0].function.name == "get_current_weather", (
        f"Server didn't return expected output\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        stream=True,
    )

    expected_outputs = ["", None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\nOpenAI client output: {r}"
        )
        if r.choices[0].delta.tool_calls:
            assert r.choices[0].delta.tool_calls[0].function.name == "get_current_weather", (
                f"Server didn't return expected output.\nOpenAI client output: {r}"
            )


@e2e_from_file("tests/e2e/default_openai_with_batching.py")
def test_e2e_openai_with_batching(openai_request_data):
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
    assert response.choices[0].message.content == (
        "Hi! It's nice to meet you. Is there something I can help you with or would you like to chat? "
    ), f"Server didn't return expected output OpenAI client output: {response}"


@e2e_from_file("tests/e2e/default_openaispec_response_format.py")
def test_openai_parity_with_response_format():
    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="lit")
    messages = [
        {
            "role": "system",
            "content": "Extract the event information.",
        },
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "calendar_event",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "date": {"type": "string"},
                    "participants": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "date", "participants"],
                "additionalProperties": "false",
            },
            "strict": "true",
        },
    }
    output = '{"name": "Science Fair", "date": "Friday", "participants": ["Alice", "Bob"]}'
    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        response_format=response_format,
    )
    assert response.choices[0].message.content == output, (
        f"Server didn't return expected output\nOpenAI client output: {response}"
    )

    response = client.chat.completions.create(
        model="lit",
        messages=messages,
        response_format=response_format,
        stream=True,
    )

    expected_outputs = [output, None]
    for r, expected_out in zip(response, expected_outputs):
        assert r.choices[0].delta.content == expected_out, (
            f"Server didn't return expected output.\nOpenAI client output: {r}"
        )


@e2e_from_file("tests/e2e/default_single_streaming.py")
def test_e2e_single_streaming():
    resp = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0}, headers=None, stream=True)
    assert resp.status_code == 200, f"Expected response to be 200 but got {resp.status_code}"

    outputs = []
    for line in resp.iter_lines():
        if line:
            outputs.append(json.loads(line.decode("utf-8")))

    assert len(outputs) == 3, "Expected 3 streamed outputs"
    assert outputs[-1] == {"output": 12.0}, "Final output doesn't match expected value"

    expected_values = [4.0, 8.0, 12.0]
    for i, output in enumerate(outputs):
        assert output["output"] == expected_values[i], f"Intermediate output {i} is not expected value"


@e2e_from_file("tests/e2e/default_openai_embedding_spec.py")
def test_openai_embedding_parity():
    client = OpenAI(
        base_url="http://127.0.0.1:8000/v1",
        api_key="lit",
    )

    model = "lit"
    input_text = "The food was delicious and the waiter was very friendly."
    input_text_list = [input_text] * 2
    response = client.embeddings.create(
        model="lit", input="The food was delicious and the waiter...", encoding_format="float"
    )
    assert response.model == model, f"Expected model to be {model} but got {response.model}"
    assert len(response.data) == 1, f"Expected 1 embeddings but got {len(response.data)}"
    assert len(response.data[0].embedding) == 768, f"Expected 768 dimensions but got {len(response.data[0].embedding)}"
    assert isinstance(response.data[0].embedding[0], float), "Expected float datatype but got something else"

    response = client.embeddings.create(model="lit", input=input_text_list, encoding_format="float")
    assert response.model == model, f"Expected model to be {model} but got {response.model}"
    assert len(response.data) == 2, f"Expected 2 embeddings but got {len(response.data)}"
    for data in response.data:
        assert len(data.embedding) == 768, f"Expected 768 dimensions but got {len(data.embedding)}"
