import logging
import time

import requests
from requests.adapters import HTTPAdapter
from rich.console import Console
from tenacity import retry, stop_after_attempt
from urllib3.util import Retry
from utils import benchmark

# Configuration
SERVER_URL = "http://0.0.0.0:8000/predict"
MAX_SPEED = 390  # Nvidia 3090

console = Console()


def create_session(pool_connections, pool_maxsize, max_retries=3):
    """Create a session object with custom connection pool settings."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.1,
    )
    adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# Initialize session with reasonable defaults
session = create_session(pool_connections=50, pool_maxsize=50)


def get_average_throughput(num_requests=100, num_samples=10):
    key = "Requests Per Second (RPS)"
    latency_key = "Latency per Request (ms)"
    metric = 0
    latency = 0

    # warmup
    benchmark(num_requests=50, concurrency_level=10, print_metrics=False)
    for i in range(num_samples):
        bnmk = benchmark(num_requests=num_requests, concurrency_level=num_requests, print_metrics=False)
        metric += bnmk[key]
        latency += bnmk[latency_key]
        if (i + 1) % 10 == 0:
            console.print(f"Completed {i + 1} samples", style="bold green")
    avg = metric / num_samples
    console.print("-" * 50, style="bold blue")
    console.print("BERT Performance Test Results", style="bold blue")
    console.print("-" * 50, style="bold blue")
    console.print("avg RPS:", avg)
    console.print("avg latency:", latency / num_samples)
    console.print("-" * 50, style="bold blue")
    return avg


@retry(stop=stop_after_attempt(10))
def main():
    for i in range(10):
        try:
            resp = session.get("http://localhost:8000/health")
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Error connecting to server: {e}")
        time.sleep(10)

    rps = get_average_throughput(100, num_samples=10)
    if rps < MAX_SPEED:
        console.print(f"\nPerformance test failed. Retrying... (Expected: {MAX_SPEED}, Got: {rps})", style="bold red")
    assert rps >= MAX_SPEED, f"Expected RPS to be greater than {MAX_SPEED}, got {rps}"


if __name__ == "__main__":
    main()
