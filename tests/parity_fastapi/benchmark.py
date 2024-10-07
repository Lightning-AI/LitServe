import base64
import concurrent.futures
import random
import time

import numpy as np
import requests
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

rand_mat = np.random.rand(2, 224, 224, 3) * 255
Image.fromarray(rand_mat[0].astype("uint8")).convert("RGB").save("image1.jpg")
Image.fromarray(rand_mat[1].astype("uint8")).convert("RGB").save("image2.jpg")

SERVER_URL = "http://127.0.0.1:{}/predict"

payloads = []
for file in ["image1.jpg", "image2.jpg"]:
    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        payloads.append(encoded_string)


def send_request(port):
    """Function to send a single request and measure the response time."""
    url = SERVER_URL.format(port)
    payload = {"image_data": random.choice(payloads)}
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    return end_time - start_time, response.status_code


def benchmark(num_requests=100, concurrency_level=100, port=8000):
    """Benchmark the ML server."""

    start_benchmark_time = time.time()  # Start benchmark timing
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, port) for _ in range(num_requests)]
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
    success_rate = status_codes.count(200) / num_requests * 100
    rps = num_requests / total_benchmark_time  # Requests per second

    # Create a dictionary with the metrics
    metrics = {
        "Total Requests": num_requests,
        "Concurrency Level": concurrency_level,
        "Total Benchmark Time (seconds)": total_benchmark_time,
        "Average Response Time (ms)": avg_time * 1000,
        "Success Rate (%)": success_rate,
        "Requests Per Second (RPS)": rps,
    }

    # Print the metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("-" * 50)

    return metrics


def run_bench(conf: dict, num_samples: int, port: int):
    num_requests = conf[device]["num_requests"]

    results = []
    for _ in range(num_samples):
        metric = benchmark(num_requests=num_requests, concurrency_level=num_requests, port=port)
        results.append(metric)
    return results[2:]  # skip warmup step
