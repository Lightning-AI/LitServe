import torch

from tests.e2e.test_e2e import e2e_from_file
from tests.parity_fastapi.benchmark import run_bench

device = "cpu" if torch.cuda.is_available() else "cuda"
device = "mps" if torch.backends.mps.is_available() else device

diff_factor = {
    "cpu": 1,
    "gpu": 2,
    "mps": 2,
}


@e2e_from_file("tests/parity_fastapi/fastapi_server.py")
def run_fastapi_benchmark(num_samples):
    return run_bench(num_samples)


@e2e_from_file("tests/parity_fastapi/ls_server.py")
def run_litserve_benchmark(num_samples):
    return run_bench(num_samples)


def test_parity_fastapi():
    key = "Requests Per Second (RPS)"
    num_samples = 1
    fastapi_df = run_fastapi_benchmark(num_samples=num_samples)
    ls_df = run_litserve_benchmark(num_samples=num_samples)
    fastapi_throughput = fastapi_df[key].mean()
    ls_throughput = ls_df[key].mean()
    factor = diff_factor[device]
    assert (
        ls_throughput > fastapi_throughput * factor
    ), f"LitServe should have larger throughput than FastAPI on {device}"


if __name__ == "__main__":
    test_parity_fastapi()
