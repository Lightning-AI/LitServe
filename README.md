<div align='center'>

<h2>
  The framework to build custom inference engines with expert control.   
  <br/>
  Engines for models, agents, MCP, multi-modal, RAG, and pipelines.
  <br/>
  No MLOps. No YAML.
</h2>    

<img alt="Lightning" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_banner2.png" width="800px" style="max-width: 100%;">

&nbsp; 
</div>

<div align='center'>
  
<pre>
✅ Build your own inference engine ✅ 2× faster than FastAPI     ✅ Agents, RAG, pipelines, more
✅ Custom logic + control          ✅ Any PyTorch model          ✅ Self-host or managed        
✅ Multi-GPU autoscaling           ✅ Batching + streaming       ✅ BYO model or vLLM           
✅ No MLOps glue code              ✅ Easy setup in Python       ✅ Serverless support          

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
    <a target="_blank" href="#host-anywhere" style="margin: 0 10px;">Hosting</a> •
    <a target="_blank" href="https://lightning.ai/docs/litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme" style="margin: 0 10px;">Docs</a>
  </div>
</div>

&nbsp;

<div align="center">
<a target="_blank" href="https://lightning.ai/docs/litserve/home/get-started?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>
</div>

&nbsp; 

# Looking for GPUs and an inference platform?
Over 340,000 developers use [Lightning Cloud](https://lightning.ai/?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) - purpose-built for PyTorch and PyTorch Lightning. 
- [GPUs](https://lightning.ai/pricing?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme) from $0.19.   
- [Clusters](https://lightning.ai/clusters?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme): frontier-grade training/inference clusters.   
- [AI Studio (vibe train)](https://lightning.ai/studios?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme): workspaces where AI helps you debug, tune and vibe train.
- [AI Studio (vibe deploy)](https://lightning.ai/studios?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme): workspaces where AI helps you optimize, and deploy models.     
- [Notebooks](https://lightning.ai/notebooks?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme): Persistent GPU workspaces where AI helps you code and analyze.
- [Inference](https://lightning.ai/deploy?utm_source=ptl_readme&utm_medium=referral&utm_campaign=ptl_readme): Deploy models as inference APIs.   

# Why LitServe?
LitServe lets you build your own inference engine. Serving engines such as vLLM serve specific model types (LLMs) with rigid abstractions. LitServe gives you the low-level control to serve any model (vision, audio, text, multi-modal), and define exactly how inference works - from batching, caching, streaming, and routing, to multi-model orchestration and custom logic. LitServe is perfect for building inference APIs, agents, chatbots, MCP servers, RAG, pipelines and more.

Self host LitServe or deploy in one-click to [Lightning AI](https://lightning.ai/litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme).

&nbsp;

# Quick start

Install LitServe via pip ([more options](https://lightning.ai/docs/litserve/home/install)):

```bash
pip install litserve
```

[Example 1](#inference-pipeline-example): Toy inference pipeline with multiple models.   
[Example 2](#agent-example): Minimal agent to fetch the news (with OpenAI API).    
([Advanced examples](#featured-examples)):    

### Inference pipeline example   

```python
import litserve as ls

# define the api to include any number of models, dbs, etc...
class InferencePipeline(ls.LitAPI):
    def setup(self, device):
        self.model1 = lambda x: x**2
        self.model2 = lambda x: x**3

    def predict(self, request):
        x = request["input"]    
        # perform calculations using both models
        a = self.model1(x)
        b = self.model2(x)
        c = a + b
        return {"output": c}

if __name__ == "__main__":
    # 12+ features like batching, streaming, etc...
    server = ls.LitServer(InferencePipeline(max_batch_size=1), accelerator="auto")
    server.run(port=8000)
```

Deploy for free to [Lightning cloud](#hosting-options) (or self host anywhere):

```bash
# Deploy for free with autoscaling, monitoring, etc...
lightning deploy server.py --cloud

# Or run locally (self host anywhere)
lightning deploy server.py
# python server.py
```

Test the server: Simulate an http request (run this on any terminal):
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"input": 4.0}'
```

### Agent example

```python
import re, requests, openai
import litserve as ls

class NewsAgent(ls.LitAPI):
    def setup(self, device):
        self.openai_client = openai.OpenAI(api_key="OPENAI_API_KEY")

    def predict(self, request):
        website_url = request.get("website_url", "https://text.npr.org/")
        website_text = re.sub(r'<[^>]+>', ' ', requests.get(website_url).text)

        # ask the LLM to tell you about the news
        llm_response = self.openai_client.chat.completions.create(
           model="gpt-3.5-turbo", 
           messages=[{"role": "user", "content": f"Based on this, what is the latest: {website_text}"}],
        )
        output = llm_response.choices[0].message.content.strip()
        return {"output": output}

if __name__ == "__main__":
    server = ls.LitServer(NewsAgent())
    server.run(port=8000)
```
Test it:
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"website_url": "https://text.npr.org/"}'
```

&nbsp;

# Key benefits   

A few key benefits:

- **Deploy any pipeline or model**: Agents, pipelines, RAG, chatbots, image models, video, speech, text, etc...
- **No MLOps glue:** LitAPI lets you build full AI systems (multi-model, agent, RAG) in one place ([more](https://lightning.ai/docs/litserve/api-reference/litapi?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)).   
- **Instant setup:** Connect models, DBs, and data in a few lines with `setup()` ([more](https://lightning.ai/docs/litserve/api-reference/litapi?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme#setup)).    
- **Optimized:** autoscaling, GPU support, and fast inference included ([more](https://lightning.ai/docs/litserve/api-reference/litserver?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)).    
- **Deploy anywhere:** self-host or one-click deploy with Lightning ([more](https://lightning.ai/docs/litserve/features/deploy-on-cloud?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)).
- **FastAPI for AI:** Built on FastAPI but optimized for AI - 2× faster with AI-specific multi-worker handling ([more]((#performance))).   
- **Expert-friendly:** Use vLLM, or build your own with full control over batching, caching, and logic ([more](https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)).    

> ⚠️ Not a vLLM or Ollama alternative out of the box. LitServe gives you lower-level flexibility to build what they do (and more) if you need it.

&nbsp;

# Featured examples    
Here are examples of inference pipelines for common model types and use cases.      
  
<pre>
<strong>Toy model:</strong>      <a target="_blank" href="#define-a-server">Hello world</a>
<strong>LLMs:</strong>           <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-llama-3-2-vision-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Llama 3.2</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">LLM Proxy server</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-ai-agent-with-tool-use?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Agent with tool use</a>
<strong>RAG:</strong>            <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">vLLM RAG (Llama 3.2)</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-1-rag-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">RAG API (LlamaIndex)</a>
<strong>NLP:</strong>            <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Hugging face</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">BERT</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-text-embedding-api-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Text embedding API</a>
<strong>Multimodal:</strong>     <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">OpenAI Clip</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-multi-modal-llm-with-minicpm?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">MiniCPM</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-phi3-5-vision-api-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Phi-3.5 Vision Instruct</a>, <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-and-chat-with-qwen2-vl-using-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Qwen2-VL</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-multi-modal-llm-with-pixtral?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Pixtral</a>
<strong>Audio:</strong>          <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Whisper</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">AudioCraft</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">StableAudio</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-noise-cancellation-api-with-deepfilternet?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Noise cancellation (DeepFilterNet)</a>
<strong>Vision:</strong>         <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Stable diffusion 2</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-image-generation-api-with-auraflow?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">AuraFlow</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-image-generation-api-with-flux?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Flux</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-super-resolution-image-api-with-aura-sr?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Image Super Resolution (Aura SR)</a>,
                <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-background-removal-api-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Background Removal</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-controlled-image-generation-api-controlnet?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Control Stable Diffusion (ControlNet)</a>
<strong>Speech:</strong>         <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Text-speech (XTTS V2)</a>, <a target="_blank" href="https://lightning.ai/bhimrajyadav/studios/deploy-a-speech-generation-api-using-parler-tts-powered-by-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Parler-TTS</a>
<strong>Classical ML:</strong>   <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-random-forest-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Random forest</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-xgboost-with-litserve?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">XGBoost</a>
<strong>Miscellaneous:</strong>  <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-an-media-conversion-api-with-ffmpeg?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">Media conversion API (ffmpeg)</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/deploy-both-pytorch-and-tensorflow-in-a-single-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">PyTorch + TensorFlow in one API</a>, <a target="_blank" href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme">LLM proxy server</a>
</pre>
</pre>

[Browse 100+ community-built templates](https://lightning.ai/studios?section=serving&utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)

&nbsp;

# Host anywhere

Self-host with full control, or deploy with [Lightning AI](https://lightning.ai/?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) in seconds with autoscaling, security, and 99.995% uptime.  
**Free tier included. No setup required. Run on your cloud**   

```bash
lightning deploy server.py --cloud
```

https://github.com/user-attachments/assets/ff83dab9-0c9f-4453-8dcb-fb9526726344

&nbsp;

# Features

<div align='center'>

| [Feature](https://lightning.ai/docs/litserve/features?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)               | Self Managed                      | [Fully Managed on Lightning](https://lightning.ai/deploy?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)         |
|----------------------------------------------------------------------|-----------------------------------|------------------------------------|
| Docker-first deployment          | ✅ DIY                             | ✅ One-click deploy                |
| Cost                             | ✅ Free (DIY)                      | ✅ Generous [free tier](https://lightning.ai/pricing?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) with pay as you go                |
| Full control                     | ✅                                 | ✅                                 |
| Use any engine (vLLM, etc.)      | ✅                                 | ✅ vLLM, Ollama, LitServe, etc.    |
| Own VPC                          | ✅ (manual setup)                  | ✅ Connect your own VPC            |
| [(2x)+ faster than plain FastAPI](#performance)                                               | ✅       | ✅                                 |
| [Bring your own model](https://lightning.ai/docs/litserve/features/full-control?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)              | ✅       | ✅                                 |
| [Build compound systems (1+ models)](https://lightning.ai/docs/litserve/home?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                 | ✅       | ✅                                 |
| [GPU autoscaling](https://lightning.ai/docs/litserve/features/gpu-inference?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                  | ✅       | ✅                                 |
| [Batching](https://lightning.ai/docs/litserve/features/batching?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                              | ✅       | ✅                                 |
| [Streaming](https://lightning.ai/docs/litserve/features/streaming?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                            | ✅       | ✅                                 |
| [Worker autoscaling](https://lightning.ai/docs/litserve/features/autoscaling?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                 | ✅       | ✅                                 |
| [Serve all models: (LLMs, vision, etc.)](https://lightning.ai/docs/litserve/examples?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)         | ✅       | ✅                                 |
| [Supports PyTorch, JAX, TF, etc...](https://lightning.ai/docs/litserve/features/full-control?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) | ✅       | ✅                                 |
| [OpenAPI compliant](https://www.openapis.org/)                                                | ✅       | ✅                                 |
| [Open AI compatibility](https://lightning.ai/docs/litserve/features/open-ai-spec?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)             | ✅       | ✅                                 |
| [MCP server support](https://lightning.ai/docs/litserve/features/mcp?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                         | ✅       | ✅                                 |
| [Asynchronous](https://lightning.ai/docs/litserve/features/async-concurrency?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                 | ✅       | ✅                                 |
| [Authentication](https://lightning.ai/docs/litserve/features/authentication?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                  | ❌ DIY   | ✅ Token, password, custom         |
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
| [20+ features](https://lightning.ai/docs/litserve/features?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme)                    | -                                 | -                                  |

</div>

&nbsp;

# Performance  
LitServe is designed for AI workloads. Specialized multi-worker handling delivers a minimum **2x speedup over FastAPI**.    

Additional features like batching and GPU autoscaling can drive performance well beyond 2x, scaling efficiently to handle more simultaneous requests than FastAPI and TorchServe.
    
Reproduce the full benchmarks [here](https://lightning.ai/docs/litserve/home/benchmarks?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) (higher is better).  

<div align="center">
  <img alt="LitServe" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_charts_v6.png" width="1000px" style="max-width: 100%;">
</div> 

These results are for image and text classification ML tasks. The performance relationships hold for other ML tasks (embedding, LLM serving, audio, segmentation, object detection, summarization etc...).   
    
***💡 Note on LLM serving:*** For high-performance LLM serving (like Ollama/vLLM), integrate [vLLM with LitServe](https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-2-rag-api?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme), use [LitGPT](https://github.com/Lightning-AI/litgpt?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme#deploy-an-llm), or build your custom vLLM-like server with LitServe. Optimizations like kv-caching, which can be done with LitServe, are needed to maximize LLM performance.

&nbsp;


# Community
LitServe is a [community project accepting contributions](https://lightning.ai/docs/litserve/community?utm_source=litserve_readme&utm_medium=referral&utm_campaign=litserve_readme) - Let's make the world's most advanced AI inference engine.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)    
