import os
import sys
from unittest.mock import patch

import pytest

from litserve.__main__ import main
from litserve.cli import _ensure_lightning_installed


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
