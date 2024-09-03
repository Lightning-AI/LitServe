import concurrent.futures
import random
import time

import gpustat
import psutil
import requests
from jsonargparse import CLI


def create_random_batch(size: int):
    result = []
    for i in range(size):
        result.append(random.choice(phrases))

    return result


# Configuration
SERVER_URL = "http://0.0.0.0:8000/predict"

session = requests.Session()

executor = None

phrases = [
    "In the midst of a bustling city, amidst the constant hum of traffic and the chatter of countless conversations, "
    "there exists a serene park where people come to escape the chaos. Children play on the swings, their laughter "
    "echoing through the air, while adults stroll along the winding paths, lost in thought. The trees, tall and"
    " majestic, provide a canopy of shade, and the flowers bloom in a riot of colors, adding to the park's charm."
    " It's a place where time seems to slow down, offering a moment of peace and reflection in an otherwise hectic"
    " world.",
    "As the sun sets over the horizon, painting the sky in hues of orange, pink, and purple, a sense of calm descends"
    " over the landscape. The day has been long and filled with activity, but now, in this magical hour, everything "
    "feels different. The birds return to their nests, their evening songs a lullaby to the world. The gentle breeze "
    "carries the scent of blooming jasmine, and the stars begin to twinkle in the darkening sky. It's a time for "
    "quiet contemplation, for appreciating the beauty of nature, and for feeling a deep connection to the universe.",
    "On a remote island, far away from the noise and pollution of modern life, there is a hidden cove where "
    "crystal-clear waters lap gently against the shore. The beach, covered in soft, white sand, is a paradise for "
    "those seeking solitude and tranquility. Palm trees sway in the breeze, their fronds rustling softly, while "
    "the sun casts a warm, golden glow over everything. Here, one can forget the worries of the world and simply "
    "exist in the moment, surrounded by the natural beauty of the island and the soothing sounds of the ocean.",
    "In an ancient forest, where the trees have stood for centuries, there is a sense of timelessness that"
    " envelops everything. The air is cool and crisp, filled with the earthy scent of moss and fallen leaves. Sunlight"
    " filters through the dense canopy, creating dappled patterns on the forest floor. Birds call to one another, and"
    " small animals scurry through the underbrush. It's a place where one can feel the weight of history, where the "
    "presence of the past is almost palpable. Walking through this forest is like stepping back in time, to a world "
    "untouched by human hands.",
    "At the edge of a vast desert, where the dunes stretch out as far as the eye can see, there is a small oasis "
    "that offers a respite from the harsh conditions. A cluster of palm trees provides shade, and a clear, cool spring"
    " bubbles up from the ground, a source of life in an otherwise barren landscape. Travelers who come across this "
    "oasis are greeted with the sight of lush greenery and the sound of birdsong. It's a place of refuge and renewal,"
    " where one can rest and recharge before continuing on their journey through the endless sands.",
    "High in the mountains, where the air is thin and the landscape is rugged, there is a hidden valley that remains"
    " largely untouched by human activity. The valley is a haven for wildlife, with streams that flow with clear,"
    " cold water and meadows filled with wildflowers. The surrounding peaks, covered in snow even in the summer, "
    "stand as silent sentinels. It's a place where one can feel a profound sense of solitude and connection to nature."
    " The beauty of the valley, with its pristine environment and abundant life, is a reminder of the importance of"
    " preserving wild places.",
    "On a quiet country road, far from the bustling cities and noisy highways, there is a small farmhouse surrounded"
    " by fields of golden wheat. The farmhouse, with its weathered wooden walls and cozy interior, is a place of warmth"
    " and hospitality. The fields, swaying gently in the breeze, are a testament to the hard work and dedication of "
    "the farmers who tend them. In the evenings, the sky is filled with stars, and the only sounds are the chirping of"
    " crickets and the distant hoot of an owl. It's a place where one can find peace and simplicity.",
    "In a quaint village, nestled in the rolling hills of the countryside, life moves at a slower pace. The cobblestone"
    " streets are lined with charming cottages, each with its own garden bursting with flowers. The village "
    "square is the heart of the community, where residents gather to catch up on news and enjoy each other's company."
    " There's a timeless quality to the village, where traditions are upheld, and everyone knows their neighbors. "
    "It's a place where one can experience the joys of small-town living, with its close-knit community and strong "
    "sense of belonging.",
    "By the side of a tranquil lake, surrounded by dense forests and towering mountains, there is a small cabin that "
    "offers a perfect retreat from the hustle and bustle of everyday life. The cabin, with its rustic charm and cozy "
    "interior, is a place to unwind and relax. The lake, calm and mirror-like, reflects the beauty of the surrounding "
    "landscape, creating a sense of peace and serenity. It's a place where one can reconnect with nature, spend quiet"
    " moments fishing or kayaking, and enjoy the simple pleasures of life in a beautiful, natural setting.",
]


def send_request():
    """Function to send a single request and measure the response time."""
    payload = {"text": random.choice(phrases)}
    start_time = time.time()
    response = session.post(SERVER_URL, json=payload)
    end_time = time.time()
    return end_time - start_time, response.status_code


def benchmark(num_requests=1000, concurrency_level=50, print_metrics=True):
    """Benchmark the ML server."""
    global executor
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


if __name__ == "__main__":
    send_request()
    CLI(benchmark)
