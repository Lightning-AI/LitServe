<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/litserveLogo.png" alt="LitGPT" height="90px"/>

# LitServe

**High-throughput serving engine for AI models, with a friendly interface and enterprise scale.**

<pre>
✅ Batching       ✅ Streaming          ✅ Auto-GPU, multi-GPU   
✅ Multi-modal    ✅ PyTorch/JAX/TF     ✅ Full control          
✅ Auth           ✅ Built on Fast API  ✅ Custom specs (Open AI)
</pre>



---

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/litserve)
![cpu-tests](https://github.com/Lightning-AI/litserve/actions/workflows/ci-testing.yml/badge.svg) [![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litserve/blob/main/LICENSE) [![Discord](https://img.shields.io/discord/1077906959069626439?style=plastic)](https://discord.gg/VptPCZkGNa)


<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="https://lightning.ai/docs/litserve/home/get-started">Get started</a> •
  <a href="https://lightning.ai/docs/litserve/examples">Examples</a> •
  <a href="#deployment-options">Deploy</a> •
  <a href="#features">Features</a> •
  <a href="https://lightning.ai/docs/litserve">Docs</a>
</p>

<p align="center">

&nbsp;
  
<a target="_blank" href="https://lightning.ai/docs/litserve/home">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>

</p>

</div>

&nbsp;

# Deploy AI models Lightning fast ⚡
LitServe is a high-throughput serving engine designed to deploy AI models at scale. It creates an API endpoint for models, managing batching, streaming, and autoscaling across CPUs and GPUs and more.

✅ **Supports all models:** LLMs, vision, time-series, etc...     
✅ **Developer friendly:** Focus on AI deployment not infrastructure.    
✅ **Minimal interface:** Zero-abstraction, hackable code-base.     
✅ **Enterprise scale:** Designed to handle large models with low latency.     
✅ **Auto GPU scaling:** Scale to multi-GPU with zero code changes.    

    
Think of LitServe as [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) for model serving (if you're familiar with Lightning) but supports every framework like PyTorch, JAX, Tensorflow and more.

&nbsp;

<div align="center" style="height: 200">
<video src="https://github.com/Lightning-AI/LitServe/assets/3640001/883b54bd-e54e-497a-8a29-0431abd77695" />
</div>

&nbsp;

# Quick start

&nbsp;

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/>
</a>
</div>

&nbsp;

Install LitServe via pip (or [advanced installs](https://lightning.ai/docs/litserve/home/install)):

```bash
pip install litserve
```
    
### Define a server    
Here's a hello world example ([explore real examples](https://lightning.ai/docs/litserve/examples)):

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
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8000)
```

Now run the server via the command-line

```bash
python server.py
```

These 2 minimal APIs allow enterprise-scale, with full control.

⚡️ LitAPI: Describes how the server will handle a request.    
⚡️ LitServer: Specify optimizations (such as batching, streaming, GPUs).
    
### Query the server
LitServe automatically generates a client when it starts. Use this client to test the server:

```bash
python client.py
```

Or query the server yourself directly
```python
import requests
response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
```

&nbsp;

# Examples
[Explore various examples](https://lightning.ai/docs/litserve/examples) that show different models deployed with LitServe:

| Example  | description | Deploy on Studios |
|---|---|---|
| [Hello world](#implement-a-server)  | Hello world model | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a> |
| [Llama 3 (8B)](https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server)  | **(LLM)** Deploy Llama 3 | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-8b-api"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a> |
| [LLM proxy server](https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model)  | **(LLM)** Routes traffic to various LLM providers for fault tolerance | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a> | 
| [ANY Hugging face model](https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly)  | **(Text)** Deploy any Hugging Face model | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a> |
| [Hugging face BERT model](https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model)  | **(Text)** Deploy model for tasks like text generation and more | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>
| [Open AI CLIP](https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve)  | **(Multimodal)** Deploy Open AI CLIP for tasks like image understanding | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>
| [Open AI Whisper](https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model)  | **(Audio)** Deploy Open AI Whisper for tasks like speech to text | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>
| [Meta AudioCraft](https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft)                     | **(Audio)** Deploy Meta's AudioCraft for music generation               | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>
| [Stable Audio](https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api)  | **(Audio)** Deploy Stable Audio for audio generation  | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>
| [Stable diffusion 2](https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2)  | **(Vision)** Deploy Stable diffusion 2 for tasks like image generation | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>
| [Text-speech (XTTS V2)](https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model)  | **(Speech)** Deploy a text to speech voice cloning API. | <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/></a>

&nbsp;

# Deployment options    
LitServe is developed by Lightning AI - An AI development platform which provides infrastructure for deploying AI models.    
Self manage your own deployments or use Lightning Studios to deploy production-grade models without cloud headaches.    

&nbsp;

<div align="center" style="height: 200">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" height="36px" alt="Deploy on Studios"/>
</div>

&nbsp;

<div align="center" style="height: 200">
<img width="640" alt="image" src="https://github.com/Lightning-AI/LitServe/assets/3640001/df873649-8b5c-4215-b340-e50da34e1181">
</div>

# Features
LitServe supports multiple advanced state-of-the-art features.

✅ [All model types: LLMs, vision, time series, etc...](https://lightning.ai/docs/litserve/examples).        
✅ [Auto-GPU scaling](https://lightning.ai/docs/litserve/features/gpu-inference).    
✅ [Authentication](https://lightning.ai/docs/litserve/features/authentication).    
✅ [Autoscaling](https://lightning.ai/docs/litserve/features/autoscaling).    
✅ [Batching](https://lightning.ai/docs/litserve/features/batching).    
✅ [Streaming](https://lightning.ai/docs/litserve/features/streaming).    
✅ [All ML frameworks: PyTorch, Jax, Tensorflow, Hugging Face...](https://lightning.ai/docs/litserve/features/full-control).        
✅ [Open AI spec](https://lightning.ai/docs/litserve/features/open-ai-spec).    
[10+ features...](https://lightning.ai/docs/litserve/features).    

&nbsp;

> [!NOTE]
> Our goal is not to jump on every hype train, but instead support features that scale
under the most demanding enterprise deployments.

# Contribute
LitServe is a [community project accepting contributions](https://lightning.ai/docs/litserve/community).    
Let's make the world's most advanced AI inference engine.

# License

litserve is released under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
See [LICENSE](https://github.com/Lightning-AI/LitServe/blob/main/LICENSE) file for details.
