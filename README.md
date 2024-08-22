<div align='center'>

# LitServe: Easily serve AI models Lightning fast ⚡    

<img alt="Lightning" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_banner2.png" width="800px" style="max-width: 100%;">

&nbsp;

<strong>Flexible, high-throughput serving engine for AI models.</strong>    
Friendly interface. Enterprise scale.
</div>

----

**LitServe** is a flexible serving engine for AI models built on FastAPI. Features like batching, streaming, and GPU autoscaling eliminate the need to rebuild a FastAPI server per model.  

LitServe is at least [2x faster](#performance) than plain FastAPI.

<div align='center'>
  
<pre>
✅ (2x)+ faster serving   ✅ Self-host or fully managed  ✅ Auto-GPU, multi-GPU   
✅ Multi-modal            ✅ PyTorch/JAX/TF              ✅ Full control          
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
Here's a hello world example ([explore real examples](#featured-examples)):

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

  <table>
  <tr>
   <td style="vertical-align: top;">
<pre>
<strong>Featured examples</strong><br>
<strong>Toy model:</strong>     <a href="#define-a-server">Hello world</a>
<strong>LLMs:</strong>          <a href="https://lightning.ai/lightning-ai/studios/deploy-a-private-llama-3-8b-api">Llama 3 (8B)</a>, <a href="https://lightning.ai/lightning-ai/studios/openai-fault-tolerant-proxy-server">LLM Proxy server</a>
<strong>NLP:</strong>           <a href="https://lightning.ai/lightning-ai/studios/deploy-any-hugging-face-model-instantly">Hugging face</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-a-hugging-face-bert-model">BERT</a>
<strong>Multimodal:</strong>    <a href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-clip-with-litserve">OpenAI Clip</a>
<strong>Audio:</strong>         <a href="https://lightning.ai/lightning-ai/studios/deploy-open-ai-s-whisper-model">Whisper</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-an-music-generation-api-with-meta-s-audio-craft">AudioCraft</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-an-audio-generation-api">StableAudio</a>
<strong>Vision:</strong>        <a href="https://lightning.ai/lightning-ai/studios/deploy-a-private-api-for-stable-diffusion-2">Stable diffusion 2</a>
<strong>Speech:</strong>        <a href="https://lightning.ai/lightning-ai/studios/deploy-a-voice-clone-api-coqui-xtts-v2-model">Text-speech (XTTS V2)</a>
<strong>Classical ML:</strong>  <a href="https://lightning.ai/lightning-ai/studios/deploy-random-forest-with-litserve">Random forest</a>, <a href="https://lightning.ai/lightning-ai/studios/deploy-xgboost-with-litserve">XGBoost</a>
<strong>Miscellaneous:</strong> <a href="https://lightning.ai/lightning-ai/studios/deploy-text-embedding-api-with-litserve">Text embedding API</a>
</pre>
    </td>
    <td style="vertical-align: top" width=800>
<!--       <a href="https://github.com/Lightning-AI/LitServe/assets/3640001/883b54bd-e54e-497a-8a29-0431abd77695" target="_blank">
        <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/video_thumbnail.png" alt="Video Thumbnail" style="width: 500px; height: auto;" />
      </a> -->
      <video align="right" src="https://github.com/user-attachments/assets/56655727-f5d7-4109-b60d-efc816e148c9" width=500 controls></video>
    </td>
  </tr>
</table>

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
✅ [Open AI compatibility](https://lightning.ai/docs/litserve/features/open-ai-spec)    

[10+ features...](https://lightning.ai/docs/litserve/features)    

**Note:** Our goal is not to jump on every hype train, but instead support features that scale
under the most demanding enterprise deployments.

&nbsp;

# Performance  
LitServe is highly optimized for parallel execution with native features optimized to scale AI workloads. Our benchmarks show that LitServe (built on FastAPI) handles more simultaneous requests than FastAPI and TorchServe (higher is better).     

Reproduce the full benchmarks [here](https://lightning.ai/docs/litserve/home/benchmarks).  

<div align="center">
  <img alt="LitServe" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/ls_charts_v6.png" width="1000px" style="max-width: 100%;">
</div> 

These results are for image and text classification ML tasks. The performance relationships hold for other ML tasks (embedding, LLM serving, audio, segmentation, object detection, summarization etc...).   
    
***💡 Note on LLM serving:*** For high-performance LLM serving (like Ollama/VLLM), use [LitGPT](https://github.com/Lightning-AI/litgpt?tab=readme-ov-file#deploy-an-llm) or build your custom VLLM-like server with LitServe. Optimizations like kv-caching, which can be done with LitServe, are needed to maximize LLM performance.

&nbsp; 

# Hosting options   
LitServe can be hosted independently on your own machines—perfect for hackers, students and developers who prefer a DIY approach.     
    
For enterprise developers or those seeking a more managed solution, [Lightning Studios](https://lightning.ai/) provides optional support with automated deployments, scaling, release management, and more, offering a robust path to low-effort, fully-managed enterprise-grade solutions.    

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
