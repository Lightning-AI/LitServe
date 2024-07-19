<div align="center">
<img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/litserveLogo.png" alt="LitGPT" height="90px"/>

# LitServe

**High-throughput serving engine for AI models.    
Friendly interface. Enterprise scale.**

<pre>
✅ Batching       ✅ Streaming          ✅ Auto-GPU, multi-GPU   
✅ Multi-modal    ✅ PyTorch/JAX/TF     ✅ Full control          
✅ Auth           ✅ Built on Fast API  ✅ Custom specs (Open AI)
</pre>



---

[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/litserve)
![cpu-tests](https://github.com/Lightning-AI/litserve/actions/workflows/ci-testing.yml/badge.svg) [![Discord](https://img.shields.io/discord/1077906959069626439?label=Get%20Help%20on%20Discord)](https://discord.gg/VptPCZkGNa)

<p align="center">
  <a href="https://lightning.ai/">Lightning AI</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#deploy-AI-models-lightning-fast-">Examples</a> •
  <a href="#deployment-options">Deploy</a> •
  <a href="#features">Features</a> •
  <a href="https://lightning.ai/docs/litserve/home/benchmarks">Benchmarks</a> •
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
LitServe is a high-throughput serving engine built on FastAPI, designed to deploy AI models at scale by creating API endpoints and managing features like batching, streaming, autoscaling across CPUs and GPUs, and more. 

With LitServe, you don't need to build a new FastAPI server from scratch for each model; it is batteries included with AI-specific techniques out of the box.     

**Key features:**
- ✅ **Supports all models:** LLMs, vision, time-series, etc...
- ✅ **All ML frameworks:** Use PyTorch, Jax, SKLearn, etc...
- ✅ **Developer friendly:** Focus on AI deployment not infrastructure.    
- ✅ **Minimal interface:** Zero-abstraction, hackable code-base.     
- ✅ **Enterprise scale:** Designed to handle large models with low latency.
- ✅ **Auto GPU scaling:** Scale to multi-GPU with zero code changes.    
- ✅ **Run anywhere:** Run yourself on any machine or fully managed on Lightning Studios.     

**Featured examples:**
| Model type         | Links                                                                                                                                       |
|------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| **Toy example** | [Hello world](#define-a-server)                                                                                                              |
| **LLMs**        | [Llama 3 (8B)](https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server), [LLM Proxy server](https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model) |
| **NLP models**  | [Any Hugging face model](https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly), [BERT model](https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model) |
| **Multimodal**  | [Open AI Clip](https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve)                                                   |
| **Audio**       | [Open AI Whisper](https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model), [Meta AudioCraft](https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft), [Stable Audio](https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api) |
| **Vision**      | [Stable diffusion 2](https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2)                                   |
| **Speech**      | [Text-speech (XTTS V2)](https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model)                               |

&nbsp;

<div align="center" style="height: 200">
<video src="https://github.com/Lightning-AI/LitServe/assets/3640001/883b54bd-e54e-497a-8a29-0431abd77695" />
</div>

&nbsp;

# Quick start

Install LitServe via pip (or [advanced installs](https://lightning.ai/docs/litserve/home/install)):

```bash
pip install litserve
```
    
### Define a server    
Here's a hello world example ([explore real examples](https://lightning.ai/docs/litserve/examples)):

```python
# server.py
import litserve as ls

# STEP 1: DEFINE A MODEL API
class SimpleLitAPI(ls.LitAPI):
    # Called once at startup. Setup models, DB connections, etc...
    def setup(self, device):
        self.model = lambda x: x**2  

    # Convert the request payload to model input.
    def decode_request(self, request):
        return request["input"] 

    # Run inference on the the model, return the output.
    def predict(self, x):
        return self.model(x) 

    # Convert the model output to a response payload.
    def encode_response(self, output):
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

The `LitAPI` class gives you full control and hackability. The `LitServer` handles advanced optimizations like batching, streaming and auto-GPU scaling.   
    
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

# Deployment options    
LitServe is developed by [Lightning AI](https://lightning.ai/) which provides infrastructure for deploying AI models. Self-manage deployments or use [Lightning Studios](https://lightning.ai/) for production-grade deployments without cloud headaches, security and 99.95% uptime SLA.     

&nbsp;

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/>
</a>
</div>

&nbsp;

| Feature                          | Self Managed                      | Fully Managed on Studios            |
|----------------------------------|-----------------------------------|-------------------------------------|
| Deployment                       | ✅ Do it yourself deployment       | ✅ One-button cloud deploy           |
| Load balancing                   | ❌                                | ✅                                  |
| Autoscaling                      | ❌                                | ✅                                  |
| Multi-machine inference          | ❌                                | ✅                                  |
| Authentication                   | ❌                                | ✅                                  |
| Own VPC                          | ❌                                | ✅                                  |
| AWS, GCP                         | ❌                                | ✅                                  |
| Use your own cloud commits       | ❌                                | ✅                                  |


&nbsp;

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

**Note:** Our goal is not to jump on every hype train, but instead support features that scale
under the most demanding enterprise deployments.

&nbsp;

# Community
LitServe is a [community project accepting contributions](https://lightning.ai/docs/litserve/community) - Let's make the world's most advanced AI inference engine.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)    
