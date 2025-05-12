import importlib.util
import subprocess
import sys


def _ensure_lightning_installed():
    if not importlib.util.find_spec("lightning_sdk"):
        print("Lightning CLI not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"])


def main():
    _ensure_lightning_installed()

    try:
        # Import the correct entry point for lightning_sdk
        from lightning_sdk.cli.entrypoint import main_cli

        # Call the lightning CLI's main function directly with our arguments
        # This bypasses the command-line entry point completely
        sys.argv[0] = "lightning"  # Make it think it was called as "lightning"
        main_cli()
    except ImportError as e:
        # If there's an issue importing or finding the right module
        print(f"Error importing lightning_sdk CLI: {e}")
        print("Please ensure `lightning-sdk` is installed correctly.")
        sys.exit(1)
