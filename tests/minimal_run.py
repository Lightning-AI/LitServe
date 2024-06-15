import subprocess
import psutil
import time
import requests


def main():
    process = subprocess.Popen(
        ["python", "tests/simple_server.py"],
    )
    print("Waiting for server to start...")
    time.sleep(5)
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json={"input": 4.0})
        assert response.status_code == 200
    except Exception:
        raise

    finally:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        process.kill()


if __name__ == "__main__":
    main()
