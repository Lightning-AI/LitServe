import pytest
from unittest.mock import patch
import os
import sys
import tempfile
import io
import docker
from litserve.docker_builder import (
    get_docker_client,
    build_docker_image_with_tempdir,
    create_tar_stream,
    build,
)


@pytest.fixture
def mock_docker_client():
    with patch("litserve.docker_builder.docker.APIClient") as mock_client:
        yield mock_client


def test_get_docker_client_success(mock_docker_client):
    mock_client_instance = mock_docker_client.return_value
    mock_client_instance.ping.return_value = True

    client = get_docker_client()
    assert client == mock_client_instance
    mock_client_instance.ping.assert_called_once()


def test_get_docker_client_import_error():
    with patch.dict(sys.modules, {"docker": None}), patch("builtins.input", return_value="no"), pytest.raises(
        SystemExit
    ):
        get_docker_client()


def test_get_docker_client_api_error(mock_docker_client):
    mock_client_instance = mock_docker_client.return_value
    mock_client_instance.ping.side_effect = docker.errors.APIError("Error")

    with pytest.raises(SystemExit):
        get_docker_client()


def test_create_tar_stream():
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_stream = create_tar_stream(tmpdir)
        assert isinstance(tar_stream, io.BytesIO)


def test_build_docker_image_with_tempdir(mock_docker_client):
    mock_client_instance = mock_docker_client.return_value
    mock_client_instance.build.return_value = [{"stream": "Step 1/1 : FROM python:3.9-slim\n"}]
    mock_client_instance.images.return_value = [{"Id": "test_image_id"}]

    with tempfile.TemporaryDirectory() as tmpdir:
        server_path = os.path.join(tmpdir, "server.py")
        requirements_path = os.path.join(tmpdir, "requirements.txt")
        with open(server_path, "w") as f:
            f.write("print('Hello, World!')")
        with open(requirements_path, "w") as f:
            f.write("")

        files = [server_path, requirements_path]
        dockerfile_content = "FROM python:3.9-slim\nCOPY . /app\n"
        image_id = build_docker_image_with_tempdir(
            client=mock_client_instance,
            dockerfile_str=dockerfile_content,
            files=files,
            tag="test_tag",
            timeout=600,
        )
        assert image_id == "test_image_id"


def test_build(mock_docker_client):
    mock_client_instance = mock_docker_client.return_value
    mock_client_instance.build.return_value = [{"stream": "Step 1/1 : FROM python:3.9-slim\n"}]
    mock_client_instance.images.return_value = [{"Id": "test_image_id"}]

    with tempfile.TemporaryDirectory() as tmpdir:
        server_path = os.path.join(tmpdir, "server.py")
        requirements_path = os.path.join(tmpdir, "requirements.txt")
        with open(server_path, "w") as f:
            f.write("print('Hello, World!')")
        with open(requirements_path, "w") as f:
            f.write("")

        with patch("litserve.docker_builder.get_docker_client", return_value=mock_client_instance):
            os.chdir(tmpdir)
            build(server_path=server_path, tag="test_tag", port=8000, timeout=600)

            mock_client_instance.build.assert_called_once()
            mock_client_instance.images.assert_called_once_with(name="test_tag")
