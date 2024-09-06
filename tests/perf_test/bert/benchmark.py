import time
import logging

import requests
from utils import benchmark
from tenacity import retry, stop_after_attempt

# Configuration
SERVER_URL = "http://0.0.0.0:8000/predict"
MAX_SPEED = 390  # Nvidia 3090

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


@retry(stop=stop_after_attempt(10))
def main():
    for i in range(10):
        try:
            resp = requests.get("http://localhost:8000/health")
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Error connecting to server: {e}")
        time.sleep(10)

    rps = get_average_throughput(100, num_samples=10)
    assert rps >= MAX_SPEED, f"Expected RPS to be greater than {MAX_SPEED}, got {rps}"


if __name__ == "__main__":
    main()
