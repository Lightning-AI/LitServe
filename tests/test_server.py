from unittest.mock import patch
from litserve.server import LitServer


def test_new_pipe(lit_server):
    pool_size = lit_server.max_pool_size
    for _ in range(pool_size):
        lit_server.new_pipe()

    assert len(lit_server.pipe_pool) == 0
    assert len(lit_server.new_pipe()) == 2


def test_index(sync_testclient):
    assert sync_testclient.get("/").text == "litserve running"


@patch("litserve.server.lifespan")
def test_device_identifiers(mock_lifespan, simple_litapi):
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
