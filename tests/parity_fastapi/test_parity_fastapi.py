import torch

from tests.e2e.test_e2e import e2e_from_file
from tests.parity_fastapi.benchmark import run_bench

device = "cpu" if torch.cuda.is_available() else "cuda"
device = "mps" if torch.backends.mps.is_available() else device

diff_factor = {
    "cpu": 1,
    "gpu": 2,
    "mps": 1,
}


@e2e_from_file("tests/parity_fastapi/fastapi_server.py")
def run_fastapi_benchmark(num_samples):
    return run_bench(num_samples)


@e2e_from_file("tests/parity_fastapi/ls_server.py")
def run_litserve_benchmark(num_samples):
    return run_bench(num_samples)


def mean(lst):
    return sum(lst) / (len(lst) + 1e-9)


def test_parity_fastapi():
    key = "Requests Per Second (RPS)"
    num_samples = 1
    fastapi_metrics = run_fastapi_benchmark(num_samples=num_samples)
    ls_metrics = run_litserve_benchmark(num_samples=num_samples)
    print("fastapi_metrics", fastapi_metrics)
    fastapi_throughput = mean([e[key] for e in fastapi_metrics])
    ls_throughput = mean([e[key] for e in ls_metrics])
    print("fastapi_throughput", fastapi_throughput)
    print("ls_throughput", ls_throughput)
    factor = diff_factor[device]
    msg = f"LitServe should have larger throughput than FastAPI on {device}\n." f"{fastapi_metrics} vs {ls_metrics}"
    assert ls_throughput > fastapi_throughput * factor, msg


if __name__ == "__main__":
    test_parity_fastapi()
