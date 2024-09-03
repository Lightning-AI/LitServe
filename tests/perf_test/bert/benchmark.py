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

    for i in range(num_samples):
        bnmk = benchmark(num_requests=num_requests, concurrency_level=num_requests)
        metric += bnmk[key]
        latency += bnmk[latency_key]
    avg = metric / num_samples
    print("avg RPS:", avg)
    print("avg latency:", latency / num_samples)
    return avg


if __name__ == "__main__":
    get_average_throughput(100, num_samples=10)
