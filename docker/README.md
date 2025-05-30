# Docker Support for LitServe

This directory contains Docker configurations for running LitServe in both CPU and GPU environments, using multi-stage builds and uv best practices.

## Available Configurations

- `Dockerfile.cpu`: For CPU-only environments
- `Dockerfile.gpu`: For GPU-enabled environments (requires NVIDIA Docker)

## Prerequisites

- Docker installed on your system (with BuildKit support)
- For GPU support: 
  - NVIDIA GPU with CUDA support
  - NVIDIA Container Toolkit (nvidia-docker2)

## Building the Images

### CPU Version
```bash
DOCKER_BUILDKIT=1 docker build -t litserve:cpu -f Dockerfile.cpu .
```

### GPU Version
```bash
DOCKER_BUILDKIT=1 docker build -t litserve:gpu -f Dockerfile.gpu .
```

## Running Containers

### CPU Version
```bash
docker run -p 8000:8000 litserve:cpu
```

### GPU Version
```bash
docker run --gpus all -p 8000:8000 litserve:gpu
```

## Environment Variables

You can customize the container behavior using these environment variables:

- `UV_COMPILE_BYTECODE`: Set to 1 (default) to compile Python bytecode during build
- `UV_LINK_MODE`: Set to 'copy' for more reliable builds
- `UV_PYTHON_PREFERENCE`: Set to 'only-managed' to use only uv-managed Python
- `PORT`: The port on which the server will listen (default: 8000)

Example with custom port:
```bash
docker run -p 9000:9000 -e PORT=9000 litserve:cpu
```

## Development

To mount your local code directory for development:

```bash
docker run -v $(pwd):/app -p 8000:8000 litserve:cpu
```

## Notes

- Uses multi-stage builds for smaller final images
- Build stage uses official uv image (ghcr.io/astral-sh/uv)
- Python 3.12 managed by uv for better compatibility
- BuildKit mount cache for faster dependency installation
- Runs as non-root user 'app' for security
- The GPU image uses CUDA 12.1 and PyTorch with CUDA 12.1 support
- The containers expose port 8000 by default
- Virtual environments are created using `uv venv` in the `.venv` directory
- Uses virtual environment in `/opt/venv` for better isolation 
