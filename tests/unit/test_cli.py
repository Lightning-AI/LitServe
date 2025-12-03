import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from litserve.__main__ import main
from litserve.cli import _ensure_lightning_installed
from litserve.cli import main as cli_main


def test_dockerize_help(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["litserve", "dockerize", "--help"])
    # argparse calls sys.exit() after printing help
    with pytest.raises(SystemExit):
        main()
    captured = capsys.readouterr()
    assert "usage:" in captured.out, "CLI did not print help message"
    assert "The path to the server file." in captured.out, "CLI did not print help message"


def test_dockerize_command(monkeypatch, capsys):
    # Assuming you have a dummy server file for testing
    dummy_server_file = "dummy_server.py"
    with open(dummy_server_file, "w") as f:
        f.write("# Dummy server file for testing\n")

    monkeypatch.setattr("sys.argv", ["litserve", "dockerize", dummy_server_file])
    main()
    captured = capsys.readouterr()
    os.remove(dummy_server_file)
    assert "Dockerfile created successfully" in captured.out, "CLI did not create Dockerfile"
    assert os.path.exists("Dockerfile"), "CLI did not create Dockerfile"


@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
def test_ensure_lightning_installed_with_pip(mock_run, mock_which, mock_find_spec, mock_is_package_installed):
    mock_is_package_installed.return_value = False
    mock_find_spec.return_value = True  # pip available
    mock_which.return_value = None  # uv not available
    _ensure_lightning_installed()
    mock_run.assert_called_once_with([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"], check=True)


@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
def test_ensure_lightning_installed_pip_preferred(mock_run, mock_which, mock_find_spec, mock_is_package_installed):
    """When both pip and uv are available, pip should be used first."""
    mock_is_package_installed.return_value = False
    mock_find_spec.return_value = True  # pip available
    mock_which.return_value = "/usr/bin/uv"  # uv also available
    _ensure_lightning_installed()
    mock_run.assert_called_once_with([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"], check=True)


@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
def test_ensure_lightning_installed_with_uv(mock_run, mock_which, mock_find_spec, mock_is_package_installed):
    mock_is_package_installed.return_value = False
    mock_find_spec.return_value = None  # pip not available
    mock_which.return_value = "/usr/bin/uv"  # uv available
    _ensure_lightning_installed()
    mock_run.assert_called_once_with(["uv", "pip", "install", "-U", "lightning-sdk"], check=True)


@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
def test_ensure_lightning_installed_fallback_to_uv(mock_run, mock_which, mock_find_spec, mock_is_package_installed):
    """When pip fails, should fall back to uv."""
    mock_is_package_installed.return_value = False
    mock_find_spec.return_value = True  # pip available
    mock_which.return_value = "/usr/bin/uv"  # uv also available
    mock_run.side_effect = [subprocess.CalledProcessError(1, "pip"), None]  # pip fails, uv succeeds
    _ensure_lightning_installed()
    assert mock_run.call_count == 2
    mock_run.assert_called_with(["uv", "pip", "install", "-U", "lightning-sdk"], check=True)


@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
def test_ensure_lightning_installed_failure(mock_run, mock_which, mock_find_spec, mock_is_package_installed):
    """When all available installers fail, should exit with error."""
    mock_is_package_installed.return_value = False
    mock_find_spec.return_value = True  # pip available
    mock_which.return_value = "/usr/bin/uv"  # uv also available
    mock_run.side_effect = subprocess.CalledProcessError(1, "install")  # both fail

    with pytest.raises(SystemExit, match="Failed to install lightning-sdk"):
        _ensure_lightning_installed()
    assert mock_run.call_count == 2  # tried both pip and uv


@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
def test_ensure_lightning_installed_no_installer_available(
    mock_run, mock_which, mock_find_spec, mock_is_package_installed
):
    """When neither pip nor uv is available, should exit with error."""
    mock_is_package_installed.return_value = False
    mock_find_spec.return_value = None  # pip not available
    mock_which.return_value = None  # uv not available

    with pytest.raises(SystemExit, match="Failed to install lightning-sdk"):
        _ensure_lightning_installed()
    mock_run.assert_not_called()  # no installer was tried


# TODO: Remove this once we have a fix for Python 3.9 and 3.10
@pytest.mark.skipif(sys.version_info[:2] in [(3, 9), (3, 10)], reason="Test fails on Python 3.9 and 3.10")
@patch("litserve.cli.is_package_installed")
@patch("litserve.cli.importlib.util.find_spec")
@patch("litserve.cli.shutil.which")
@patch("subprocess.run")
@patch("builtins.__import__")
def test_cli_main_lightning_not_installed(mock_import, mock_run, mock_which, mock_find_spec, mock_is_package_installed):
    # Create a mock for the lightning_sdk module and its components
    mock_lightning_sdk = MagicMock()
    mock_lightning_sdk.cli.entrypoint.main_cli = MagicMock()

    # Configure __import__ to return our mock when lightning_sdk is imported
    def side_effect(name, *args, **kwargs):
        if name == "lightning_sdk.cli.entrypoint":
            return mock_lightning_sdk
        return __import__(name, *args, **kwargs)

    mock_import.side_effect = side_effect
    mock_find_spec.return_value = True  # pip available
    mock_which.return_value = None  # uv not available

    # Test when lightning_sdk is not installed but gets installed dynamically
    mock_is_package_installed.side_effect = [False, True]  # First call returns False, second call returns True
    test_args = ["lightning", "run", "app", "app.py"]

    with patch.object(sys, "argv", test_args):
        cli_main()

    mock_run.assert_called_once_with([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"], check=True)


@pytest.mark.skipif(sys.version_info[:2] in [(3, 9), (3, 10)], reason="Test fails on Python 3.9 and 3.10")
@patch("importlib.util.find_spec")
@patch("builtins.__import__")
def test_cli_main_import_error(mock_import, mock_find_spec, capsys):
    # Set up the mock to raise ImportError specifically for lightning_sdk import
    def import_mock(name, *args, **kwargs):
        if name == "lightning_sdk.cli.entrypoint":
            raise ImportError("Module not found")
        return __import__(name, *args, **kwargs)

    mock_import.side_effect = import_mock

    # Mock find_spec to return True so we attempt the import
    mock_find_spec.return_value = True
    test_args = ["lightning", "deploy", "api", "app.py"]

    with patch.object(sys, "argv", test_args):  # noqa: SIM117
        with pytest.raises(SystemExit) as excinfo:
            cli_main()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Error importing lightning_sdk CLI" in captured.out
