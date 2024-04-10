from multiprocessing import Pipe
from unittest.mock import patch, MagicMock
from litserve.server import inference_worker
from litserve.server import LitServer


def test_new_pipe(lit_server):
    pool_size = lit_server.max_pool_size
    for _ in range(pool_size):
        lit_server.new_pipe()

    assert len(lit_server.pipe_pool) == 0
    assert len(lit_server.new_pipe()) == 2


def test_dispose_pipe(lit_server):
    for i in range(lit_server.max_pool_size + 10):
        lit_server.dispose_pipe(*Pipe())
    assert len(lit_server.pipe_pool) == lit_server.max_pool_size


def test_index(sync_testclient):
    assert sync_testclient.get("/").text == "litserve running"


@patch("litserve.server.lifespan")
def test_device_identifiers(lifespan_mock, simple_litapi):
    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    assert server.device_identifiers("cpu", 1) == ["cpu:1"]
    assert server.device_identifiers("cpu", [1, 2]) == ["cpu:1", "cpu:2"]

    server = LitServer(simple_litapi, accelerator="cpu", devices=1, timeout=10)
    assert server.app.devices == ["cpu"]

    server = LitServer(simple_litapi, accelerator="cuda", devices=1, timeout=10)
    assert server.app.devices == [["cuda:0"]]

    server = LitServer(simple_litapi, accelerator="cuda", devices=[1, 2], timeout=10)
    # [["cuda:1"], ["cuda:2"]]
    assert server.app.devices[0][0] == "cuda:1"
    assert server.app.devices[1][0] == "cuda:2"


@patch("litserve.server.run_batched_loop")
@patch("litserve.server.run_single_loop")
def test_inference_worker(mock_single_loop, mock_batched_loop):
    inference_worker(*[MagicMock()] * 5, max_batch_size=2, batch_timeout=0)
    mock_batched_loop.assert_called_once()

    inference_worker(*[MagicMock()] * 5, max_batch_size=1, batch_timeout=0)
    mock_single_loop.assert_called_once()
