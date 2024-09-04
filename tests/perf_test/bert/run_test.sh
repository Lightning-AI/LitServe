#!/bin/bash

# Function to clean up server process
cleanup() {
    pkill -f "python tests/perf_test/bert/server.py"
}

# Trap script exit to run cleanup
trap cleanup EXIT

# Start the server in the background and capture its PID
python tests/perf_test/bert/server.py &
SERVER_PID=$!

echo "Server started with PID $SERVER_PID"

# Run your benchmark script
echo "Preparing to run benchmark.py..."

# Wait till localhost:8000/health returns "ok" using wget
while [ "$(wget -qO- localhost:8000/health)" != "ok" ]; do
    sleep 1
done

export PYTHONPATH=$PWD && python tests/perf_test/bert/benchmark.py

# Check if benchmark.py exited successfully
if [ $? -ne 0 ]; then
    echo "benchmark.py failed to run successfully."
    exit 1
else
    echo "benchmark.py ran successfully."
fi
