import os
import sys
from unittest.mock import patch

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


@patch("importlib.util.find_spec")
@patch("subprocess.check_call")
def test_ensure_lightning_installed(mock_check_call, mock_find_spec):
    mock_find_spec.return_value = False
    _ensure_lightning_installed()
    mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"])


@patch("importlib.util.find_spec")
@patch("subprocess.check_call")
@patch("lightning_sdk.cli.entrypoint.main_cli")
def test_cli_main_lightning_not_installed(mock_main_cli, mock_check_call, mock_find_spec):
    # Test when lightning_sdk is not installed but gets installed dynamically
    mock_find_spec.side_effect = [False, True]  # First call returns False, second call returns True
    test_args = ["lightning", "run", "app", "app.py"]

    with patch.object(sys, "argv", test_args):
        cli_main()

    # Verify pip install was called
    mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", "-U", "lightning-sdk"])
    # Verify the lightning CLI was called after installation
    mock_main_cli.assert_called_once()


@patch("importlib.util.find_spec")
@patch("lightning_sdk.cli.entrypoint.main_cli", side_effect=ImportError("Module not found"))
def test_cli_main_import_error(mock_main_cli, mock_find_spec, capsys):
    # Test handling of ImportError
    mock_find_spec.return_value = True
    test_args = ["lightning", "run", "app", "app.py"]

    with patch.object(sys, "argv", test_args):  # noqa: SIM117
        with pytest.raises(SystemExit) as excinfo:
            cli_main()

    # Verify the right exit code
    assert excinfo.value.code == 1

    # Check error message
    captured = capsys.readouterr()
    assert "Error importing lightning_sdk CLI" in captured.out
    assert "Please ensure lightning-sdk is installed correctly" in captured.out
