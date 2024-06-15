import subprocess
import psutil
import time
import json
import urllib.request


def main():
    process = subprocess.Popen(
        ["python", "tests/simple_server.py"],
    )
    print("Waiting for server to start...")
    time.sleep(5)
    try:
        url = "http://127.0.0.1:8000/predict"
        data = json.dumps({"input": 4.0}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        response = urllib.request.urlopen(request)
        status_code = response.getcode()
        assert status_code == 200
    except Exception:
        raise

    finally:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        process.kill()


if __name__ == "__main__":
    main()
