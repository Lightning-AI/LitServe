import torch
import requests

from benchmark import run_bench
import psutil
import subprocess
import time

from functools import wraps

conf = {
    "cpu": {"num_requests": 8},
    "mps": {"num_requests": 8},
    "cuda": {"num_requests": 16},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

diff_factor = {
    "cpu": 1,
    "cuda": 100,
    "mps": 1,
}


def run_python_script(filename):
    def decorator(test_fn):
        @wraps(test_fn)
        def wrapper(*args, **kwargs):
            process = subprocess.Popen(
                ["python", filename],
            )
            print("Waiting for server to start...")
            time.sleep(10)

            try:
                return test_fn(*args, **kwargs)
            except Exception:
                raise
            finally:
                print("Killing the server")
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                process.kill()

        return wrapper

    return decorator


def try_health(port):
    for i in range(10):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health")
            if response.status_code == 200:
                return
        except Exception:
            pass


@run_python_script("tests/parity_fastapi/fastapi-server.py")
def run_fastapi_benchmark(num_samples):
    port = 8001
    try_health(port)
    return run_bench(conf, num_samples, port)


@run_python_script("tests/parity_fastapi/ls-server.py")
def run_litserve_benchmark(num_samples):
    port = 8000
    try_health(port)
    return run_bench(conf, num_samples, port)


def mean(lst):
    return sum(lst) / len(lst)


def main():
    key = "Requests Per Second (RPS)"
    num_samples = 6
    fastapi_metrics = run_fastapi_benchmark(num_samples=num_samples)
    ls_metrics = run_litserve_benchmark(num_samples=num_samples)
    fastapi_throughput = mean([e[key] for e in fastapi_metrics])
    ls_throughput = mean([e[key] for e in ls_metrics])
    factor = diff_factor[device]
    msg = f"LitServe should have higher throughput than FastAPI on {device}. {ls_throughput} vs {fastapi_throughput}"
    assert ls_throughput > fastapi_throughput + factor, msg


if __name__ == "__main__":
    main()
