# Docker Support for LitServe

Production-ready Docker configurations for LitServe API servers.

## Quick Start

### Build Images

```bash
# CPU version
docker build -t litserve:cpu -f docker/Dockerfile.cpu .

# GPU version (requires NVIDIA Docker)
docker build -t litserve:gpu -f docker/Dockerfile.gpu .
```

### Run Your Server

**Mount your server file:**
```bash
# CPU
docker run -v $(pwd)/server.py:/app/server.py -p 8000:8000 litserve:cpu python server.py

# GPU
docker run --gpus all -v $(pwd)/server.py:/app/server.py -p 8000:8000 litserve:gpu python server.py
```

**Or build with your server included:**
```dockerfile
FROM litserve:cpu
COPY server.py /app/
CMD ["python", "server.py"]
```

## Prerequisites

- **CPU**: Docker with BuildKit
- **GPU**: NVIDIA Docker + CUDA-compatible GPU

## Key Features

- ✅ Optimized multi-stage builds
- ✅ Non-root user for security  
- ✅ Python 3.12 + uv package manager
- ✅ Layer caching for fast rebuilds
- ✅ Production-ready configurations

## Environment Variables

- `PORT`: Server port (default: 8000)
- `UV_COMPILE_BYTECODE=1`: Faster Python execution
- `UV_SYSTEM_PYTHON=1`: Use system Python

---

💡 **Need help?** Check the [LitServe documentation](https://github.com/Lightning-AI/litserve) for server examples.
