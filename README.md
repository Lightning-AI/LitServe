<div align='center'>

<h1>
  Deploy agents, RAG, models, pipelines and more.
  <br/>
  Without learning MLOps.
</h1>    

<img alt="Lightning" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_banner2.png" width="800px" style="max-width: 100%;">

&nbsp; 
</div>

**Most AI inference tools are built around single-model APIs with rigid abstractions**. They lock you into serving one model per server, with no way to customize internals like batching, caching, or kernels. This makes it hard to build full systems like RAG or agents without stitching together multiple services. The result is complex MLOps orchestration, slower iteration, and bloated infrastructure.   

**LitServe flips this paradigm**: Write full AI pipelines, not just models, in clean, extensible Python. Built on FastAPI but optimized for AI workloads, LitServe supports multi-model serving, streaming, batching, and custom logic - all from a single server. Deploy in one click with autoscaling, monitoring, and zero infrastructure overhead. Or run it self-hosted with full control and no lock-in.

LitServe is at least [2x faster](#performance) than plain FastAPI due to AI-specific multi-worker handling.    

<div align='center'>
  
<pre>
✅ (2x)+ faster serving  ✅ Easy to use               ✅ LLMs, non LLMs and more
✅ Bring your own model  ✅ PyTorch/JAX/TF/...        ✅ Built on FastAPI       
✅ GPU autoscaling       ✅ Batching, Streaming       ✅ Self-host or ⚡️ managed
✅ Inference pipeline    ✅ Integrate with vLLM, etc  ✅ Serverless             
   
</pre>

<div align='center'>

[![PyPI Downloads](https://static.pepy.tech/badge/litserve)](https://pepy.tech/projects/litserve)
[![Discord](https://img.shields.io/discord/1077906959069626439?label=Get%20help%20on%20Discord)](https://discord.gg/WajDThKAur)
![cpu-tests](https://github.com/Lightning-AI/litserve/actions/workflows/ci-testing.yml/badge.svg)
[![codecov](https://codecov.io/gh/Lightning-AI/litserve/graph/badge.svg?token=SmzX8mnKlA)](https://codecov.io/gh/Lightning-AI/litserve)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)

</div>
</div>
<div align="center">
  <div style="text-align: center;">
    <a target="_blank" href="#quick-start" style="margin: 0 10px;">Quick start</a> •
    <a target="_blank" href="#featured-examples" style="margin: 0 10px;">Examples</a> •
    <a target="_blank" href="#features" style="margin: 0 10px;">Features</a> •
    <a target="_blank" href="#performance" style="margin: 0 10px;">Performance</a> •
    <a target="_blank" href="#hosting-options" style="margin: 0 10px;">Hosting</a> •
    <a target="_blank" href="https://lightning.ai/docs/litserve" style="margin: 0 10px;">Docs</a>
  </div>
</div>

&nbsp;

<div align="center">
<a target="_blank" href="https://lightning.ai/docs/litserve/home/get-started">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>
</div>

&nbsp; 

# Quick start

Install LitServe via pip ([more options](https://lightning.ai/docs/litserve/home/install)):

```bash
pip install litserve
```
    
### Define a server    
This toy example with 2 models (inference pipeline) shows LitServe's flexibility ([see real examples](#featured-examples)):    

```python
import litserve as ls

# define the api to include any number of models, dbs, etc...
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        self.model1 = lambda x: x**2
        self.model2 = lambda x: x**3

    def decode_request(self, request):
        # get inputs to /predict
        return request["input"]

    def predict(self, x):
        # perform calculations using both models
        a = self.model1(x)
        b = self.model2(x)
        c = a + b
        return {"output": c}

    def encode_response(self, output):
        # package outputs from /predict
        return {"output": output} 

if __name__ == "__main__":
    # 12+ features like batching, streaming, etc...
    server = ls.LitServer(SimpleLitAPI(max_batch_size=1), accelerator="auto")
    server.run(port=8000)
```

Now deploy for free to [Lightning cloud](#hosting-options) (or self host anywhere):

```bash
# Deploy for free with autoscaling, monitoring, etc...
lightning deploy server.py --cloud

# Or run locally (self host anywhere)
lightning deploy server.py
# python server.py
```

### Test the server
Simulate an http request (run this on any terminal):
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": 4.0}'
```

### LLM serving
LitServe isn’t *just* for LLMs like vLLM or Ollama; it serves any AI model with full control over internals ([learn more](https://lightning.ai/docs/litserve/features/serve-llms)).    
For easy LLM serving, integrate [vLLM with LitServe](https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api), or use [LitGPT](https://github.com/Lightning-AI/litgpt?tab=readme-ov-file#deploy-an-llm) (built on LitServe). 

```
litgpt serve microsoft/phi-2
```

### Summary
- LitAPI lets you easily build complex AI systems with one or more models ([docs](https://lightning.ai/docs/litserve/api-reference/litapi)).
- Use the setup method for one-time tasks like connecting models, DBs, and loading data ([docs](https://lightning.ai/docs/litserve/api-reference/litapi#setup)).        
- LitServer handles optimizations like batching, GPU autoscaling, streaming, etc... ([docs](https://lightning.ai/docs/litserve/api-reference/litserver)).
- Self host on your machines or create a fully managed deployment with Lightning ([learn more](https://lightning.ai/docs/litserve/features/deploy-on-cloud)).

[Learn how to make this server 200x faster](https://lightning.ai/docs/litserve/home/speed-up-serving-by-200x).    

&nbsp;

# Featured examples    
Here are examples of inference pipelines for common model types and use cases.      
  
<pre>
<strong>Toy model:</strong>      <a target="_blank" href="#define-a-server">Hello world</a>
<strong>LLMs:</strong>           <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-llama-3-2-vision-with-litserve">Llama 3.2</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server">LLM Proxy server</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-ai-agent-with-tool-use">Agent with tool use</a>
<strong>RAG:</strong>            <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api">vLLM RAG (Llama 3.2)</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-1-rag-api">RAG API (LlamaIndex)</a>
<strong>NLP:</strong>            <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly">Hugging face</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model">BERT</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-text-embedding-api-with-litserve">Text embedding API</a>
<strong>Multimodal:</strong>     <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve">OpenAI Clip</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-multi-modal-llm-with-minicpm">MiniCPM</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-phi3-5-vision-api-with-litserve">Phi-3.5 Vision Instruct</a>, <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-and-chat-with-qwen2-vl-using-litserve">Qwen2-VL</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-multi-modal-llm-with-pixtral">Pixtral</a>
<strong>Audio:</strong>          <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model">Whisper</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft">AudioCraft</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api">StableAudio</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-noise-cancellation-api-with-deepfilternet">Noise cancellation (DeepFilterNet)</a>
<strong>Vision:</strong>         <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2">Stable diffusion 2</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-image-generation-api-with-auraflow">AuraFlow</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-image-generation-api-with-flux">Flux</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-super-resolution-image-api-with-aura-sr">Image Super Resolution (Aura SR)</a>,
                <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-background-removal-api-with-litserve">Background Removal</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-controlled-image-generation-api-controlnet">Control Stable Diffusion (ControlNet)</a>
<strong>Speech:</strong>         <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model">Text-speech (XTTS V2)</a>, <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-a-speech-generation-api-using-parler-tts-powered-by-litserve">Parler-TTS</a>
<strong>Classical ML:</strong>   <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-random-forest-with-litserve">Random forest</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-xgboost-with-litserve">XGBoost</a>
<strong>Miscellaneous:</strong>  <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-media-conversion-api-with-ffmpeg">Media conversion API (ffmpeg)</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-both-pytorch-and-tensorflow-in-a-single-api">PyTorch + TensorFlow in one API</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server">LLM proxy server</a>
</pre>
</pre>

[Browse 100+ community-built templates](https://lightning.ai/studios?section=serving)

&nbsp;


# Hosting options   
Self host LitServe anywhere or deploy to your favorite cloud via [Lightning AI](http://lightning.ai/deploy).

https://github.com/user-attachments/assets/ff83dab9-0c9f-4453-8dcb-fb9526726344

Self-hosting is ideal for hackers, students, and DIY developers while fully managed hosting is ideal for enterprise developers needing easy autoscaling, security, release management, and 99.995% uptime and observability.

*Note:* Lightning offers a generous free tier for developers.

To host on [Lightning AI](https://lightning.ai/deploy), simply run the command, login and choose the cloud of your choice.
```bash
lightning deploy server.py --cloud
```

&nbsp;

## Features

<div align='center'>

| [Feature](https://lightning.ai/docs/litserve/features)               | Self Managed                      | [Fully Managed on Lightning](https://lightning.ai/deploy)         |
|----------------------------------------------------------------------|-----------------------------------|------------------------------------|
| Docker-first deployment          | ✅ DIY                             | ✅ One-click deploy                |
| Cost                             | ✅ Free (DIY)                      | ✅ Generous [free tier](https://lightning.ai/pricing) with pay as you go                |
| Full control                     | ✅                                 | ✅                                 |
| Use any engine (vLLM, etc.)      | ✅                                 | ✅ vLLM, Ollama, LitServe, etc.    |
| Own VPC                          | ✅ (manual setup)                  | ✅ Connect your own VPC            |
| [(2x)+ faster than plain FastAPI](#performance)                                               | ✅       | ✅                                 |
| [Bring your own model](https://lightning.ai/docs/litserve/features/full-control)              | ✅       | ✅                                 |
| [Build compound systems (1+ models)](https://lightning.ai/docs/litserve/home)                 | ✅       | ✅                                 |
| [GPU autoscaling](https://lightning.ai/docs/litserve/features/gpu-inference)                  | ✅       | ✅                                 |
| [Batching](https://lightning.ai/docs/litserve/features/batching)                              | ✅       | ✅                                 |
| [Streaming](https://lightning.ai/docs/litserve/features/streaming)                            | ✅       | ✅                                 |
| [Worker autoscaling](https://lightning.ai/docs/litserve/features/autoscaling)                 | ✅       | ✅                                 |
| [Serve all models: (LLMs, vision, etc.)](https://lightning.ai/docs/litserve/examples)         | ✅       | ✅                                 |
| [Supports PyTorch, JAX, TF, etc...](https://lightning.ai/docs/litserve/features/full-control) | ✅       | ✅                                 |
| [OpenAPI compliant](https://www.openapis.org/)                                                | ✅       | ✅                                 |
| [Open AI compatibility](https://lightning.ai/docs/litserve/features/open-ai-spec)             | ✅       | ✅                                 |
| [Authentication](https://lightning.ai/docs/litserve/features/authentication)                  | ❌ DIY   | ✅ Token, password, custom         |
| GPUs                             | ❌ DIY                             | ✅ 8+ GPU types, H100s from $1.75  |
| Load balancing                   | ❌                                 | ✅ Built-in                        |
| Scale to zero (serverless)       | ❌                                 | ✅ No machine runs when idle       |
| Autoscale up on demand           | ❌                                 | ✅ Auto scale up/down              |
| Multi-node inference             | ❌                                 | ✅ Distribute across nodes         |
| Use AWS/GCP credits              | ❌                                 | ✅ Use existing cloud commits      |
| Versioning                       | ❌                                 | ✅ Make and roll back releases     |
| Enterprise-grade uptime (99.95%) | ❌                                 | ✅ SLA-backed                      |
| SOC2 / HIPAA compliance          | ❌                                 | ✅ Certified & secure              |
| Observability                    | ❌                                 | ✅ Built-in, connect 3rd party tools|
| CI/CD ready                      | ❌                                 | ✅ Lightning SDK                   |
| 24/7 enterprise support          | ❌                                 | ✅ Dedicated support               |
| Cost controls & audit logs       | ❌                                 | ✅ Budgets, breakdowns, logs       |
| Debug on GPUs                    | ❌                                 | ✅ Studio integration              |
| [20+ features](https://lightning.ai/docs/litserve/features)                    | -                                 | -                                  |

</div>

&nbsp;

# Performance  
LitServe is designed for AI workloads. Specialized multi-worker handling delivers a minimum **2x speedup over FastAPI**.    

Additional features like batching and GPU autoscaling can drive performance well beyond 2x, scaling efficiently to handle more simultaneous requests than FastAPI and TorchServe.
    
Reproduce the full benchmarks [here](https://lightning.ai/docs/litserve/home/benchmarks) (higher is better).  

<div align="center">
  <img alt="LitServe" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_charts_v6.png" width="1000px" style="max-width: 100%;">
</div> 

These results are for image and text classification ML tasks. The performance relationships hold for other ML tasks (embedding, LLM serving, audio, segmentation, object detection, summarization etc...).   
    
***💡 Note on LLM serving:*** For high-performance LLM serving (like Ollama/vLLM), integrate [vLLM with LitServe](https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api), use [LitGPT](https://github.com/Lightning-AI/litgpt?tab=readme-ov-file#deploy-an-llm), or build your custom vLLM-like server with LitServe. Optimizations like kv-caching, which can be done with LitServe, are needed to maximize LLM performance.

&nbsp;


# Community
LitServe is a [community project accepting contributions](https://lightning.ai/docs/litserve/community) - Let's make the world's most advanced AI inference engine.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)    
