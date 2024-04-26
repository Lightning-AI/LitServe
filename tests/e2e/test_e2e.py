import os
import subprocess
import time


def test_default_batching():
    process = subprocess.Popen(
        ["python", "tests/e2e/default_batching.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    time.sleep(5)
    assert os.path.exists("client.py"), f"Expected client file to be created at {os.getcwd()} after starting the server"
    output = subprocess.run("python client.py", shell=True, capture_output=True, text=True).stdout
    assert '{"output":16.0}' in output, "tests/simple_server.py didn't return expected output"
    os.remove("client.py")
    process.kill()
