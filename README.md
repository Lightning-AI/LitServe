<div align='center'>

# LitServe: Easily serve AI models Lightning fast ⚡    

<img alt="Lightning" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_banner2.png" width="800px" style="max-width: 100%;">

&nbsp;

<strong>High-throughput serving engine for AI models.</strong>    
Flexible. Friendly interface. Enterprise scale.
</div>

----

**LitServe** is a flexible serving engine for AI models built on FastAPI. Features like batching, streaming, and GPU autoscaling eliminate the need to rebuild a FastAPI server per model.  

LitServe is at least [2x faster](#performance) than plain FastAPI.

<div align='center'>
  
<pre>
✅ (2x)+ faster serving   ✅ Self-host or fully managed  ✅ GPU autoscaling  
✅ Multi-modal            ✅ PyTorch/JAX/TF              ✅ OpenAPI compliant
✅ Batching               ✅ Built on Fast API           ✅ Streaming        
</pre>

<div align='center'>

[![Discord](https://img.shields.io/discord/1077906959069626439?label=Get%20help%20on%20Discord)](https://discord.gg/VptPCZkGNa)
![cpu-tests](https://github.com/Lightning-AI/litserve/actions/workflows/ci-testing.yml/badge.svg)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)

</div>
</div>
<div align="center">
  <div style="text-align: center;">
    <a href="#quick-start" style="margin: 0 10px;">Quick start</a> •
    <a href="#featured-examples" style="margin: 0 10px;">Examples</a> •
    <a href="#features" style="margin: 0 10px;">Features</a> •
    <a href="#performance" style="margin: 0 10px;">Performance</a> •
    <a href="#hosting-options" style="margin: 0 10px;">Hosting</a> •
    <a href="https://lightning.ai/docs/litserve" style="margin: 0 10px;">Docs</a>
  </div>
</div>

&nbsp;

<div align="center">
<a target="_blank" href="https://lightning.ai/docs/litserve/home">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/get-started-badge.svg" height="36px" alt="Get started"/>
</a>
</div>

&nbsp; 

# Quick start

Install LitServe via pip ([other install options](https://lightning.ai/docs/litserve/home/install)):

```bash
pip install litserve
```
    
### Define a server    
Here's a toy example with 2 models that highlights the flexibility ([explore real examples](#featured-examples)):

```python
# server.py
import litserve as ls

# STEP 1: DEFINE A MODEL API
class SimpleLitAPI(ls.LitAPI):
    def setup(self, device):
        # setup is called once at startup. Build a compound AI system (1+ models), connect DBs, load data, etc...
        self.model1 = lambda x: x**2
        self.model2 = lambda x: x**3

    def decode_request(self, request):
        # Convert the request payload to model input.
        return request["input"] 

    def predict(self, x):
        # Run inference on the the AI system, return the output.
        squared = self.model1(x)
        cubed = self.model2(x)
        output = squared + cubed
        return {"output": output}

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

`LitAPI` class gives full control and hackability.    
`LitServer` handles optimizations like batching, auto-GPU scaling, etc...      
    
### Query the server

Use the automatically generated LitServe client:

```bash
python client.py
```

<details>
  <summary>Write a custom client</summary>

```python
import requests
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"input": 4.0}
)
```
</details>

&nbsp;


# Featured examples    
Use LitServe to deploy any model or AI service: (Gen AI, classical ML, embedding servers, LLMs, vision, audio, multi-modal systems, etc...)       

<div align='center'>
  <div width='200px'>
        <video src="https://github.com/user-attachments/assets/5e73549a-bc0f-47a9-9d9c-5b54389be5de" width='200px' controls></video>    
  </div>
</div>


<pre>
<strong>Featured examples</strong><br>
<strong>Toy model:</strong>      <a href="#define-a-server">Hello world</a>
<strong>LLMs:</strong>           <a href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-8b-api">Llama 3 (8B)</a>, <a href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server">LLM Proxy server</a>
<strong>NLP:</strong>            <a href="https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly">Hugging face</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model">BERT</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-text-embedding-api-with-litserve">Text embedding API</a>
<strong>Multimodal:</strong>     <a href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve">OpenAI Clip</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-a-multi-modal-llm-with-minicpm">MiniCPM</a>, <a href="https://lightning.ai/lightning-ai/studios/run-meta-s-chameleon-30b">Chameleon 30B</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-phi3-5-vision-api-with-litserve">Phi-3.5 Vision Instruct</a>
<strong>Audio:</strong>          <a href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model">Whisper</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft">AudioCraft</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api">StableAudio</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-a-noise-cancellation-api-with-deepfilternet">Noise cancellation (DeepFilterNet)</a>
<strong>Vision:</strong>         <a href="https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2">Stable diffusion 2</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-an-image-generation-api-with-auraflow">AuraFlow</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-an-image-generation-api-with-flux">Flux</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-a-super-resolution-image-api-with-aura-sr">Image super resolution (Aura SR)</a>
<strong>Speech:</strong>         <a href="https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model">Text-speech (XTTS V2)</a>
<strong>Classical ML:</strong>   <a href="https://lightning.ai/lightning-ai/studios/deploy-random-forest-with-litserve">Random forest</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-xgboost-with-litserve">XGBoost</a>
<strong>Miscellaneous:</strong>  <a href="https://lightning.ai/lightning-ai/studios/deploy-an-media-conversion-api-with-ffmpeg">Media conversion API (ffmpeg)</a>
</pre>

[Browse 100s of community-built templates](https://lightning.ai/studios?section=serving).

&nbsp;

# Features
LitServe supports multiple advanced state-of-the-art features.

✅ [(2x)+ faster serving than plain FastAPI](#performance)          
✅ [Self host on your own machines](https://lightning.ai/docs/litserve/features/hosting-methods#host-on-your-own)    
✅ [Host fully managed on Lightning AI](https://lightning.ai/docs/litserve/features/hosting-methods#host-on-lightning-studios)  
✅ [Serve all models: LLMs, vision, time series, etc...](https://lightning.ai/docs/litserve/examples)        
✅ [Auto-GPU scaling](https://lightning.ai/docs/litserve/features/gpu-inference)    
✅ [Authentication](https://lightning.ai/docs/litserve/features/authentication)    
✅ [Autoscaling](https://lightning.ai/docs/litserve/features/autoscaling)    
✅ [Batching](https://lightning.ai/docs/litserve/features/batching)    
✅ [Streaming](https://lightning.ai/docs/litserve/features/streaming)    
✅ [Scale to zero (serverless)](https://lightning.ai/docs/litserve/features/streaming)    
✅ [All ML frameworks: PyTorch, Jax, Tensorflow, Hugging Face...](https://lightning.ai/docs/litserve/features/full-control)        
✅ [OpenAPI compliant](https://www.openapis.org/)          
✅ [Open AI compatibility](https://lightning.ai/docs/litserve/features/open-ai-spec)    

[10+ features...](https://lightning.ai/docs/litserve/features)    

**Note:** Our goal is not to jump on every hype train, but instead support features that scale
under the most demanding enterprise deployments.

&nbsp;

# Performance  
LitServe is designed for AI workloads. Specialized multi-worker handling delivers a minimum **2x speedup over FastAPI**.    

Additional features like batching and GPU autoscaling can drive performance well beyond 2x, scaling efficiently to handle more simultaneous requests than FastAPI and TorchServe.
    
Reproduce the full benchmarks [here](https://lightning.ai/docs/litserve/home/benchmarks) (higher is better).  

<div align="center">
  <img alt="LitServe" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_charts_v6.png" width="1000px" style="max-width: 100%;">
</div> 

These results are for image and text classification ML tasks. The performance relationships hold for other ML tasks (embedding, LLM serving, audio, segmentation, object detection, summarization etc...).   
    
***💡 Note on LLM serving:*** For high-performance LLM serving (like Ollama/VLLM), use [LitGPT](https://github.com/Lightning-AI/litgpt?tab=readme-ov-file#deploy-an-llm) or build your custom VLLM-like server with LitServe. Optimizations like kv-caching, which can be done with LitServe, are needed to maximize LLM performance.

&nbsp; 

# Hosting options   
LitServe can be hosted independently on your own machines or fully managed via Lightning Studios.

Self-hosting is ideal for hackers, students, and DIY developers, while fully managed hosting is ideal for enterprise developers needing easy autoscaling, security, release management, and 99.995% uptime and observability.   

&nbsp;

<div align="center">
<a target="_blank" href="https://lightning.ai/lightning-ai/studios/litserve-hello-world">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/deploy-on-studios.svg" alt="Open In Studio"/>
</a>
</div>

&nbsp;

<div align='center'>
  
| Feature                          | Self Managed                      | Fully Managed on Studios            |
|----------------------------------|-----------------------------------|-------------------------------------|
| Deployment                       | ✅ Do it yourself deployment      | ✅ One-button cloud deploy          |
| Load balancing                   | ❌                                | ✅                                  |
| Autoscaling                      | ❌                                | ✅                                  |
| Scale to zero                    | ❌                                | ✅                                  |
| Multi-machine inference          | ❌                                | ✅                                  |
| Authentication                   | ❌                                | ✅                                  |
| Own VPC                          | ❌                                | ✅                                  |
| AWS, GCP                         | ❌                                | ✅                                  |
| Use your own cloud commits       | ❌                                | ✅                                  |

</div>

&nbsp;

# Community
LitServe is a [community project accepting contributions](https://lightning.ai/docs/litserve/community) - Let's make the world's most advanced AI inference engine.

💬 [Get help on Discord](https://discord.com/invite/XncpTy7DSt)    
📋 [License: Apache 2.0](https://github.com/Lightning-AI/litserve/blob/main/LICENSE)    
