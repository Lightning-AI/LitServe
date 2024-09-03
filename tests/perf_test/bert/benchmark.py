import requests
from simple_benchmark import benchmark

# Configuration
SERVER_URL = "http://0.0.0.0:8000/predict"

session = requests.Session()


def get_average_throughput(num_requests=100, num_samples=10):
    key = "Requests Per Second (RPS)"
    latency_key = "Latency per Request (ms)"
    metric = 0
    latency = 0

    # warmup
    benchmark(num_requests=50, concurrency_level=10)
    for i in range(num_samples):
        bnmk = benchmark(num_requests=num_requests, concurrency_level=num_requests)
        metric += bnmk[key]
        latency += bnmk[latency_key]
    avg = metric / num_samples
    print("avg RPS:", avg)
    print("avg latency:", latency / num_samples)
    return avg


if __name__ == "__main__":
    rps = get_average_throughput(100, num_samples=10)
    assert rps >= 350, f"Expected RPS to be greater than 350, got {rps}"
