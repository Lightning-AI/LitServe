#! /usr/bin/env python3
# Run performance test as a job

import sys
from datetime import datetime

from lightning_sdk import Studio


def main(gh_run_id: str = ""):
    if not gh_run_id:
        gh_run_id = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    print("Creating studio...")
    s = Studio(f"litserve-perf-run{gh_run_id}", "oss-litserve", org="lightning-ai", create_ok=True)

    try:
        s.upload_folder("./")

        print("Starting studio...")
        s.start()
        s.run("pip install . -U -r _requirements/test.txt")
        s.run("pip install gpustat wget -r _requirements/perf.txt")

        print("Running BERT test...")
        s.run("bash tests/perf_test/bert/run_test.sh")

    finally:
        s.stop()


if __name__ == "__main__":
    main(sys.argv[1])
