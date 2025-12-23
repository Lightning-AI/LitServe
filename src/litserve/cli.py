import importlib.util
import shutil
import subprocess
import sys

from litserve.utils import is_package_installed


def _ensure_lightning_installed():
    """Ensure lightning-sdk is installed, attempting auto-installation if needed."""
    if is_package_installed("lightning_sdk"):
        return

    print("Lightning CLI not found. Installing lightning-sdk...")

    # Build list of available installers (pip first as it respects the active environment)
    installers = []
    if importlib.util.find_spec("pip"):
        installers.append([sys.executable, "-m", "pip"])
    if shutil.which("uv"):
        installers.append(["uv", "pip"])

    for installer in installers:
        try:
            subprocess.run([*installer, "install", "-U", "lightning-sdk"], check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    sys.exit("Failed to install lightning-sdk. Run: pip install lightning-sdk")


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
