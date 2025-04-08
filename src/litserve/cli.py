import importlib.util
import subprocess
import sys


def _ensure_lightning_installed():
    if not importlib.util.find_spec("lightning_sdk"):
        print("Lightning CLI not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"])


def main():
    _ensure_lightning_installed()

    # Forward CLI arguments to the real lightning command
    cli_args = sys.argv[1:]
    subprocess.run(["lightning"] + cli_args)
