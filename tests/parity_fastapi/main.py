import torch

from benchmark import run_bench
import psutil
import subprocess
import time

from functools import wraps


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


conf = {
    "cpu": {"num_requests": 8},
    "mps": {"num_requests": 8},
    "cuda": {"num_requests": 8},
}

device = "cpu" if torch.cuda.is_available() else "cuda"
device = "mps" if torch.backends.mps.is_available() else device

diff_factor = {
    "cpu": 1,
    "gpu": 2,
    "mps": 1,
}


def try_health():
    for i in range(10):
        try:
            import requests

            response = requests.get("http://127.0.0.1:8000/health")
            if response.status_code == 200:
                return
        except Exception:
            pass


@run_python_script("tests/parity_fastapi/fastapi_server.py")
def run_fastapi_benchmark(num_samples):
    try_health()
    return run_bench(conf, num_samples)


@run_python_script("tests/parity_fastapi/ls_server.py")
def run_litserve_benchmark(num_samples):
    try_health()
    return run_bench(conf, num_samples)


def mean(lst):
    return sum(lst) / len(lst)


def main():
    key = "Requests Per Second (RPS)"
    num_samples = 5
    fastapi_metrics = run_fastapi_benchmark(num_samples=num_samples)
    ls_metrics = run_litserve_benchmark(num_samples=num_samples)
    fastapi_throughput = mean([e[key] for e in fastapi_metrics])
    ls_throughput = mean([e[key] for e in ls_metrics])
    factor = diff_factor[device]
    msg = f"LitServe should have higher throughput than FastAPI on {device}. {fastapi_throughput} vs {ls_throughput}"
    assert ls_throughput > fastapi_throughput * factor, msg


if __name__ == "__main__":
    main()
