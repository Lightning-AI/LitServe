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

## Running a Server

The containers use `uv run` as the entrypoint. There are two ways to run your server:

### Option 1: Mount your server file
```bash
# CPU version
docker run -v $(pwd)/server.py:/app/server.py -p 8000:8000 litserve:cpu server.py

# GPU version
docker run --gpus all -v $(pwd)/server.py:/app/server.py -p 8000:8000 litserve:gpu server.py
```

### Option 2: Build image with server file
Create a new Dockerfile:
```dockerfile
FROM litserve:cpu  # or litserve:gpu for GPU support
COPY server.py /app/server.py
CMD ["server.py"]
```

Then build and run:
```bash
docker build -t myapp .
docker run -p 8000:8000 myapp
```

### Development Mode with Hot Reload
For development with automatic reloading on code changes:
```bash
# CPU version
docker run -v $(pwd):/app -p 8000:8000 litserve:cpu server.py --reload

# GPU version
docker run --gpus all -v $(pwd):/app -p 8000:8000 litserve:gpu server.py --reload
```

### Custom Server Configuration
```bash
# Set host and port
docker run -v $(pwd)/server.py:/app/server.py -p 9000:9000 litserve:cpu server.py --host 0.0.0.0 --port 9000

# With environment variables
docker run -v $(pwd)/server.py:/app/server.py -p 8000:8000 -e WORKERS=4 litserve:cpu server.py

# With custom config file
docker run -v $(pwd):/app -p 8000:8000 litserve:cpu server.py --config config.yaml
```

## Project Structure Example
```
your-project/
├── Dockerfile              # Optional: for building with server included
├── server.py              # Your LitServe server file
├── config.yaml            # Optional: server configuration
└── docker/
    ├── Dockerfile.cpu     # Base CPU image
    ├── Dockerfile.gpu     # Base GPU image
    └── README.md          # This file
```

## Environment Variables

You can customize the container behavior using these environment variables:

- `UV_COMPILE_BYTECODE`: Set to 1 (default) to compile Python bytecode during build
- `UV_LINK_MODE`: Set to 'copy' for more reliable builds
- `UV_PYTHON_PREFERENCE`: Set to 'only-managed' to use only uv-managed Python
- `UV_NO_CACHE`: Set to 1 (default in container) to disable uv caching
- `PORT`: The port on which the server will listen (default: 8000)

Example with custom port:
```bash
docker run -p 9000:9000 -e PORT=9000 litserve:cpu
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
