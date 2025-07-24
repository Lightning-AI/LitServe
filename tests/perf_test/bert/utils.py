import concurrent.futures
import random
import time

import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from tests.perf_test.bert.data import phrases


def create_random_batch(size: int):
    result = []
    for i in range(size):
        result.append(random.choice(phrases))

    return result


# Configuration
SERVER_URL = "http://0.0.0.0:8000/predict"


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

executor = None


def send_request():
    """Function to send a single request and measure the response time."""
    payload = {"text": random.choice(phrases)}
    start_time = time.time()
    response = session.post(SERVER_URL, json=payload)
    end_time = time.time()
    return end_time - start_time, response.status_code


def benchmark(num_requests=1000, concurrency_level=50, print_metrics=True):
    """Benchmark the ML server."""
    import gpustat

    global executor, session

    # Update session if concurrency level changes
    if session.adapters["http://"].poolmanager.connection_pool_kw["maxsize"] < concurrency_level:
        session = create_session(pool_connections=min(concurrency_level, 100), pool_maxsize=min(concurrency_level, 100))

    if executor is None:
        print("creating executor")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level)

    if executor._max_workers < concurrency_level:
        print("updating executor")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level)

    start_benchmark_time = time.time()  # Start benchmark timing
    futures = [executor.submit(send_request) for _ in range(num_requests)]
    response_times = []
    status_codes = []

    for future in concurrent.futures.as_completed(futures):
        response_time, status_code = future.result()
        response_times.append(response_time)
        status_codes.append(status_code)

    end_benchmark_time = time.time()  # End benchmark timing
    total_benchmark_time = end_benchmark_time - start_benchmark_time  # Time in seconds

    # Analysis
    total_time = sum(response_times)  # Time in seconds
    avg_time = total_time / num_requests  # Time in seconds
    avg_latency_per_request = (total_time / num_requests) * 1000  # Convert to milliseconds
    success_rate = status_codes.count(200) / num_requests * 100
    rps = num_requests / total_benchmark_time  # Requests per second

    # Calculate throughput per concurrent user in requests per second
    successful_requests = status_codes.count(200)
    throughput_per_user = (successful_requests / total_benchmark_time) / concurrency_level  # Requests per second

    # Create a dictionary with the metrics
    metrics = {
        "Total Requests": num_requests,
        "Concurrency Level": concurrency_level,
        "Total Benchmark Time (seconds)": total_benchmark_time,
        "Average Response Time (ms)": avg_time * 1000,
        "Success Rate (%)": success_rate,
        "Requests Per Second (RPS)": rps,
        "Latency per Request (ms)": avg_latency_per_request,
        "Throughput per Concurrent User (requests/second)": throughput_per_user,
    }
    try:
        gpu_stats = gpustat.GPUStatCollection.new_query()
        metrics["GPU Utilization"] = sum([gpu.utilization for gpu in gpu_stats.gpus])  # / len(gpu_stats.gpus)
    except Exception:
        metrics["GPU Utilization"] = -1
    metrics["CPU Usage"] = psutil.cpu_percent(0.5)

    # Print the metrics
    if print_metrics:
        for key, value in metrics.items():
            print(f"{key}: {value}")
        print("-" * 50)

    return metrics
