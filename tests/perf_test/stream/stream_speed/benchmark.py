"""Consume 10K tokens from the stream endpoint and measure the speed."""

import logging
import time

import requests
from tenacity import retry, stop_after_attempt

logger = logging.getLogger(__name__)
# Configuration
SERVER_URL = "http://0.0.0.0:8000/predict"
TOTAL_TOKENS = 10000
EXPECTED_TTFT = 0.005  # time to first token

# tokens per second
MAX_SPEED = 3600  # 3600 on GitHub CI, 10000 on M3 Pro

session = requests.Session()


def speed_test():
    start = time.time()
    resp = session.post(SERVER_URL, stream=True, json={"input": 1})
    num_tokens = 0
    ttft = None  # time to first token
    for line in resp.iter_lines():
        if not line:
            continue
        if ttft is None:
            ttft = time.time() - start
            print(f"Time to first token: {ttft}")
            assert ttft < EXPECTED_TTFT, f"Expected time to first token to be less than 0.1 seconds but got {ttft}"
        num_tokens += 1
    end = time.time()
    resp.raise_for_status()
    assert num_tokens == TOTAL_TOKENS, f"Expected {TOTAL_TOKENS} tokens, got {num_tokens}"
    speed = num_tokens / (end - start)
    return {"speed": speed, "time": end - start}


@retry(stop=stop_after_attempt(10))
def main():
    for i in range(10):
        try:
            resp = requests.get("http://localhost:8000/health")
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Error connecting to server: {e}")
        time.sleep(10)
    data = speed_test()
    speed = data["speed"]
    print(data)
    assert speed >= MAX_SPEED, f"Expected streaming speed to be greater than {MAX_SPEED}, got {speed}"


if __name__ == "__main__":
    main()
