import time
import concurrent.futures
import statistics
import requests
import asyncio
import aiohttp

import random

phrases = [
    "In the midst of a bustling city, amidst the constant hum of traffic and the chatter of countless conversations, there exists a serene park where people come to escape the chaos. Children play on the swings, their laughter echoing through the air, while adults stroll along the winding paths, lost in thought. The trees, tall and majestic, provide a canopy of shade, and the flowers bloom in a riot of colors, adding to the park's charm. It's a place where time seems to slow down, offering a moment of peace and reflection in an otherwise hectic world.",
    "As the sun sets over the horizon, painting the sky in hues of orange, pink, and purple, a sense of calm descends over the landscape. The day has been long and filled with activity, but now, in this magical hour, everything feels different. The birds return to their nests, their evening songs a lullaby to the world. The gentle breeze carries the scent of blooming jasmine, and the stars begin to twinkle in the darkening sky. It's a time for quiet contemplation, for appreciating the beauty of nature, and for feeling a deep connection to the universe.",
    "On a remote island, far away from the noise and pollution of modern life, there is a hidden cove where crystal-clear waters lap gently against the shore. The beach, covered in soft, white sand, is a paradise for those seeking solitude and tranquility. Palm trees sway in the breeze, their fronds rustling softly, while the sun casts a warm, golden glow over everything. Here, one can forget the worries of the world and simply exist in the moment, surrounded by the natural beauty of the island and the soothing sounds of the ocean.",
    "In an ancient forest, where the trees have stood for centuries, there is a sense of timelessness that envelops everything. The air is cool and crisp, filled with the earthy scent of moss and fallen leaves. Sunlight filters through the dense canopy, creating dappled patterns on the forest floor. Birds call to one another, and small animals scurry through the underbrush. It's a place where one can feel the weight of history, where the presence of the past is almost palpable. Walking through this forest is like stepping back in time, to a world untouched by human hands.",
    "At the edge of a vast desert, where the dunes stretch out as far as the eye can see, there is a small oasis that offers a respite from the harsh conditions. A cluster of palm trees provides shade, and a clear, cool spring bubbles up from the ground, a source of life in an otherwise barren landscape. Travelers who come across this oasis are greeted with the sight of lush greenery and the sound of birdsong. It's a place of refuge and renewal, where one can rest and recharge before continuing on their journey through the endless sands.",
    "High in the mountains, where the air is thin and the landscape is rugged, there is a hidden valley that remains largely untouched by human activity. The valley is a haven for wildlife, with streams that flow with clear, cold water and meadows filled with wildflowers. The surrounding peaks, covered in snow even in the summer, stand as silent sentinels. It's a place where one can feel a profound sense of solitude and connection to nature. The beauty of the valley, with its pristine environment and abundant life, is a reminder of the importance of preserving wild places.",
    "On a quiet country road, far from the bustling cities and noisy highways, there is a small farmhouse surrounded by fields of golden wheat. The farmhouse, with its weathered wooden walls and cozy interior, is a place of warmth and hospitality. The fields, swaying gently in the breeze, are a testament to the hard work and dedication of the farmers who tend them. In the evenings, the sky is filled with stars, and the only sounds are the chirping of crickets and the distant hoot of an owl. It's a place where one can find peace and simplicity.",
    "In a quaint village, nestled in the rolling hills of the countryside, life moves at a slower pace. The cobblestone streets are lined with charming cottages, each with its own garden bursting with flowers. The village square is the heart of the community, where residents gather to catch up on news and enjoy each other's company. There's a timeless quality to the village, where traditions are upheld, and everyone knows their neighbors. It's a place where one can experience the joys of small-town living, with its close-knit community and strong sense of belonging.",
    "By the side of a tranquil lake, surrounded by dense forests and towering mountains, there is a small cabin that offers a perfect retreat from the hustle and bustle of everyday life. The cabin, with its rustic charm and cozy interior, is a place to unwind and relax. The lake, calm and mirror-like, reflects the beauty of the surrounding landscape, creating a sense of peace and serenity. It's a place where one can reconnect with nature, spend quiet moments fishing or kayaking, and enjoy the simple pleasures of life in a beautiful, natural setting.",
    "In the heart of a bustling city, there is a hidden garden that provides a peaceful escape from the noise and activity of urban life. The garden, filled with a variety of plants and flowers, is a haven for birds and butterflies. A small fountain in the center adds to the tranquil atmosphere, its gentle sound masking the distant hum of traffic. Benches are scattered throughout, offering places to sit and reflect. It's a place where city dwellers can find a moment of calm and connect with nature, even in the midst of a busy metropolis.",
]


def create_random_batch(size: int):
    result = []
    for i in range(size):
        result.append(random.choice(phrases))

    return result


URL = "http://0.0.0.0:8000/predict"
session = requests.Session()


def send_request(url, text):
    start_time = time.time()
    response = session.post(url, data={"text": text})
    end_time = time.time()
    return (end_time - start_time), response


async def send_async_request(session, url, text):
    start_time = time.time()
    async with session.post(url, data={"text": text}) as response:
        end_time = time.time()
        return (end_time - start_time), await response.text()


def warmup_server(url, data, num_requests=5):
    for _ in range(num_requests):
        send_request(url, data)

async def warmup_server_async(url, data, num_requests=5):
    async with aiohttp.ClientSession() as session:
        tasks = [send_async_request(session, url, data) for _ in range(num_requests)]
        await asyncio.gather(*tasks)


def benchmark_server(data, num_requests, concurrency=8):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, URL, data[i]) for i in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        response_times = [e[0] for e in results]
        responses = [e[1] for e in results]

    t1 = time.time()
    total_time = t1 - t0
    average_time = sum(response_times) / len(response_times)
    median_time = statistics.median(response_times)

    print(f"Total benchmark time: {total_time*1000:.2f} ms")
    print(f"Average response time: {average_time * 1000:.2f} ms")
    print(f"Median response time: {median_time * 1000:.2f} ms")

    return total_time, responses


async def async_benchmark_server(data, num_requests, concurrency=8):
    t0 = time.time()
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [send_async_request(session, URL, data[i]) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        response_times = [e[0] for e in results]
        responses = [e[1] for e in results]

    t1 = time.time()
    total_time = t1 - t0
    average_time = sum(response_times) / len(response_times)
    median_time = statistics.median(response_times)

    print(f"Total benchmark time: {total_time*1000:.2f} ms")
    print(f"Average response time: {average_time * 1000:.2f} ms")
    print(f"Median response time: {median_time * 1000:.2f} ms")

    return total_time, responses


def main():
    times = []
    warmup_data = "Warmup request text"

    # Perform warmup requests
    # warmup_server(URL, warmup_data)
    asyncio.run(warmup_server_async(URL, warmup_data))

    for i in range(10):
        n = 32
        data = create_random_batch(n)

        # Synchronous benchmark
        # total_time, responses = benchmark_server(num_requests=n, concurrency=n, data=data)
        # Asynchronous benchmark
        total_time, responses = asyncio.run(async_benchmark_server(num_requests=n, concurrency=n, data=data))

        times.append(total_time)

    average_time = sum(times) / len(times)
    median_time = statistics.median(times)

    print(f"Average total benchmark time: {average_time * 1000:.2f} ms")
    print(f"Median total benchmark time: {median_time * 1000:.2f} ms")


if __name__ == "__main__":
    main()
