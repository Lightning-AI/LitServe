<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/litserveLogo.png" alt="LitGPT" height="90px"/>

&nbsp;

# LitServe

**High-throughput serving engine for AI models**

✅ Batching &nbsp; &nbsp;  ✅ Streaming &nbsp; &nbsp;  ✅ Multi-GPU &nbsp; &nbsp;  ✅ PyTorch/JAX/TF &nbsp; &nbsp;  ✅ Full control &nbsp; &nbsp;  ✅ Auth

---


<p align="center">

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

</p>

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/litserve)
![cpu-tests](https://github.com/Lightning-AI/litserve/actions/workflows/ci-testing.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litserve/blob/main/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)


<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#install-litserve">Install</a> •
  <a href="#get-started">Get started</a> •
  <a href="#features">Features</a>
</p>

</div>

&nbsp;

# Deploy AI models Lightning fast ⚡
LitServe is a high-throughput serving engine for deploying AI models at scale. LitServe generates an API endpoint for a model, handles batching, streaming, autoscaling across CPU/GPUs and more.

Why we wrote LitServe:

1. Work with any model: LLMs, vision, time-series, etc...
3. We wanted a zero abstraction, minimal, hackable code-base without bloat.
5. Built for enterprise scale (not demos, etc...).
6. Easy enough for researchers, scalable and hackable for engineers.
2. Work on any hardware (GPU/TPU) automatically.
5. Let you focus on model performance, not the serving boilerplate.

Think of LitServe as PyTorch Lightning for model serving (if you're familiar with Lightning) but supports every framework like PyTorch, JAX, Tensorflow and more.

<div align="center">
    <img src="https://github.com/Lightning-AI/litserve/assets/3640001/4a4a5028-1e64-46f3-b0db-ef5b3f636655" height="160px">
</div>

Run the hello world demo here:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

&nbsp;

# Install LitServe
Install LitServe via pip:

```bash
pip install litserve
```

<details>
  <summary>Advanced install options</summary>
&nbsp;

Install the main branch:

```bash
pip install https://github.com/Lightning-AI/litserve/main.zip
```
&nbsp;

Install from source:

```bash
git clone https://github.com/Lightning-AI/litserve
cd litserve
pip install -e '.[all]'
```

</details>


# Get started
LitServe is an inference server for AI/ML models that is minimal and highly scalable.

It has 2 simple, minimal APIs - LitAPI and LitServer.

## Implement a server
Here's a hello world example:

<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

```python
# server.py
import litserve as ls

# STEP 1: DEFINE YOUR MODEL API
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # Setup the model so it can be called in `predict`.
        self.model = lambda x: x**2

    def decode_request(self, request):
        # Convert the request payload to your model input.
        return request["input"]

    def predict(self, x):
        # Run the model on the input and return the output.
        return self.model(x)

    def encode_response(self, output):
        # Convert the model output to a response payload.
        return {"output": output}


# STEP 2: START THE SERVER
if __name__ == "__main__":
    api = SimpleLitAPI()
    server = ls.LitServer(api, accelerator="gpu")
    server.run(port=8000)
```

Now run the server via the command-line

```bash
python server.py
```

## Use the server
LitServe automatically generates a client when it starts. Use this client to test the server:

```bash
python client.py
```

Or ping the server yourself directly
```python
import requests
response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
```

The server expects the client to send a `POST` to the `/predict` URL with a JSON payload.
The way the payload is structured is up to the implementation of the `LitAPI` subclass.

# Features
LitServe supports multiple advanced state-of-the-art features.

| Feature  | description  |
|---|---|
| Accelerators  | CPU, GPU, Multi-GPU  |
| Model types  | LLMs, Vision, Time series, any model type...  |
| ML frameworks  | PyTorch, Jax, Tensorflow, numpy, etc...  |
| Batching | ✅ |
| API authentication | ✅ |
| Full request/response control | ✅ |
| Automatic schema validation | ✅ |
| Handle timeouts | ✅ |
| Handle disconnects | ✅ |
| Streaming | in progress... |

> [!NOTE]
> Our goal is not to jump on every hype train, but instead support features that scale
under the most demanding enterprise deployments.

## Feature details

Explore each feature in detail:

<details>
  <summary>Automatic schema validation</summary>

&nbsp;

Define the request and response as [Pydantic models](https://docs.pydantic.dev/latest/),
to automatically validate the request.

```python
from pydantic import BaseModel


class PredictRequest(BaseModel):
    input: float


class PredictResponse(BaseModel):
    output: float


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        self.model = lambda x: x**2

    def decode_request(self, request: PredictRequest) -> float:
        return request.input

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output: float) -> PredictResponse:
        return PredictResponse(output=output)


if __name__ == "__main__":
    api = SimpleLitAPI()
    server = LitServer(api, accelerator="cpu")
    server.run(port=8888)
```

</details>

<details>
  <summary>Serve on GPUs</summary>

&nbsp;

`LitServer` has the ability to coordinate serving from multiple GPUs.

For example, running the API server on a 4-GPU machine, with a PyTorch model served by each GPU:

```python
from fastapi import Request, Response

from litserve import LitAPI, LitServer

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.linear.weight.data.fill_(2.0)
        self.linear.bias.data.fill_(1.0)

    def forward(self, x):
        return self.linear(x)


class SimpleLitAPI(LitAPI):
    def setup(self, device):
        # move the model to the correct device
        # keep track of the device for moving data accordingly
        self.model = Linear().to(device)
        self.device = device

    def decode_request(self, request: Request):
        # get the input and create a 1D tensor on the correct device
        content = request["input"]
        return torch.tensor([content], device=self.device)

    def predict(self, x):
        # the model expects a batch dimension, so create it
        return self.model(x[None, :])

    def encode_response(self, output) -> Response:
        # float will take the output value directly onto CPU memory
        return {"output": float(output)}


if __name__ == "__main__":
    # accelerator="cuda", devices=4 will lead to 4 workers serving the
    # model from "cuda:0", "cuda:1", "cuda:2", "cuda:3" respectively
    server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=4)
    server.run(port=8000)
```

The `devices` variable can also be an array specifying what device id to
run the model on:

```python
server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=[0, 3])
```

Last, you can run multiple copies of the same model from the same device,
if the model is small. The following will load two copies of the model on
each of the 4 GPUs:

```python
server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=4, workers_per_device=2)
```

</details>

<details>
  <summary>Timeouts and disconnections</summary>

&nbsp;

The server will remove a queued request if the client requesting it disconnects.

You can configure a timeout (in seconds) after which clients will receive a `504` HTTP
response (Gateway Timeout) indicating that their request has timed out.

For example, this is how you can configure the server with a timeout of 30 seconds per response.

```python
server = LitServer(SimpleLitAPI(), accelerator="cuda", devices=4, timeout=30)
```

This is useful to avoid requests queuing up beyond the ability of the server to respond.

</details>

<details>
  <summary>Use API key authentication</summary>

&nbsp;

In order to secure the API behind an API key, just define the env var when
starting the server

```bash
LIT_SERVER_API_KEY=supersecretkey python main.py
```

Clients are expected to auth with the same API key set in the `X-API-Key` HTTP header.

</details>
&nbsp;


# Examples

Learn and use these examples to quickly get started for serving the model of your choice.

<details>
    <summary>Serve Llama 3</summary>

You can serve Llama 3 and stream chat response to client.

```python
from typing import Generator, List
import json
from pathlib import Path
from typing import Dict, Any, Optional, Literal
from litgpt.utils import check_valid_checkpoint_dir

import lightning as L
import torch
from litserve import LitAPI, LitServer

from litgpt.model import GPT
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer
from litgpt.generate.base import generate, next_token
from litgpt.prompts import load_prompt_style, has_prompt_style, PromptStyle
from litgpt.utils import load_checkpoint, CLI, get_default_supported_precision
from pydantic import BaseModel


class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 50


class LlamaAPI(LitAPI):
    def __init__(
        self,
        checkpoint_dir: Path,
        precision: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.precision = precision

    def setup(self, device: str) -> None:
        # Setup the model so it can be called in `predict`.
        config = Config.from_file(self.checkpoint_dir / "model_config.yaml")
        device = torch.device(device)
        torch.set_float32_matmul_precision("high")

        precision = self.precision or get_default_supported_precision(training=False)

        fabric = L.Fabric(
            accelerator=device.type,
            devices=1
            if device.type == "cpu"
            else [device.index],  # TODO: Update once LitServe supports "auto"
            precision=precision,
        )
        checkpoint_path = self.checkpoint_dir / "lit_model.pth"
        self.tokenizer = Tokenizer(self.checkpoint_dir)
        self.prompt_style = (
            load_prompt_style(self.checkpoint_dir)
            if has_prompt_style(self.checkpoint_dir)
            else PromptStyle.from_config(config)
        )
        with fabric.init_module(empty_init=True):
            model = GPT(config)
        with fabric.init_tensor():
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()

        self.model = fabric.setup_module(model)
        load_checkpoint(fabric, self.model, checkpoint_path)
        self.device = fabric.device

    def decode_request(self, request: PromptRequest) -> Any:
        # Convert the request payload to your model input.
        prompt = request.prompt
        prompt = self.prompt_style.apply(prompt)
        encoded = self.tokenizer.encode(prompt, device=self.device)
        return encoded, request

    @torch.inference_mode()
    def generate_iter(
        self,
        model: GPT,
        prompt: torch.Tensor,
        max_returned_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

        The implementation of this function is modified from A. Karpathy's nanoGPT.

        Args:
            model: The model to use.
            prompt: Tensor of shape (T) with indices of the prompt sequence.
            max_returned_tokens: The maximum number of tokens to return (given plus generated).
            temperature: Scales the predicted logits by 1 / temperature.
            top_k: If specified, only sample among the tokens with the k highest probabilities.
            eos_id: If specified, stop generating any more token once the <eos> token is triggered.
        """
        T = prompt.size(0)
        assert max_returned_tokens > T
        if model.max_seq_length < max_returned_tokens - 1:
            # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
            # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
            # not support it to avoid negatively impacting the overall speed
            raise NotImplementedError(
                f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}"
            )

        device = prompt.device
        tokens = [prompt]
        input_pos = torch.tensor([T], device=device)
        token = next_token(
            model,
            torch.arange(0, T, device=device),
            prompt.view(1, -1),
            temperature=temperature,
            top_k=top_k,
        ).clone()
        tokens.append(token)
        for _ in range(2, max_returned_tokens - T + 1):
            token = next_token(
                model,
                input_pos,
                token.view(1, -1),
                temperature=temperature,
                top_k=top_k,
            ).clone()
            if token == eos_id:
                break
            input_pos = input_pos.add_(1)
            yield token

    def predict(self, x: List) -> Generator:
        # Run the model on the input and return the output.
        inputs, request = x
        prompt_length = inputs.size(0)
        max_returned_tokens = prompt_length + request.max_new_tokens

        y_iter = self.generate_iter(
            self.model,
            inputs,
            max_returned_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            eos_id=self.tokenizer.eos_id,
        )
        for token in y_iter:
            yield token

        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()

    def encode_response(self, outputs: Generator) -> Generator:
        # Convert the model output to a response payload.
        for output in outputs:
            decoded_output = self.tokenizer.decode(output)
            yield json.dumps({"output": decoded_output})


if __name__ == "__main__":
    # 1. Download Llama 3:
    # litgpt download --repo_id meta-llama/Meta-Llama-3-8B-Instruct

    # 2. Run server
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Meta-Llama-3-8B-Instruct")
    check_valid_checkpoint_dir(checkpoint_dir, model_filename="lit_model.pth")

    server = LitServer(
        LlamaAPI(checkpoint_dir=checkpoint_dir),
        accelerator="cuda",
        devices=1,
        stream=True,
    )

    server.run(port=8000)
```

You can stream response with a Python client as follows:

```python
import requests
import json

url = "http://127.0.0.1:8000/stream-predict"
prompt = "Write a Python code to sort a linkedlist in reverse order. Please be short."
resp = requests.post(
    url,
    json={
        "prompt": prompt,
        "max_new_tokens": 200,
    },
    stream=True,
)
for chunk in resp.iter_content(chunk_size=4000):
    if chunk:
        msg = json.loads(chunk.decode("utf-8"))["output"]
        print(msg, end="")
```

</details>


## License

litserve is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
See LICENSE file for details.


# Run Tests

You can run tests locally using `pytest` to verify that all the tests pass after making any changes.

First, you need to install the test dependencies:

```shell
pip install -r _requirements/test.txt
```

Then, run pytest in your terminal as follows:

```shell
pytest tests
```
