from tests.e2e.test_e2e import e2e_from_file
from tests.parity_fastapi.benchmark import run_bench


@e2e_from_file("tests/parity_fastapi/fastapi_server.py")
def run_fastapi_benchmark():
    return run_bench(1)


@e2e_from_file("tests/parity_fastapi/ls_server.py")
def run_litserve_benchmark():
    return run_bench(1)


def test_parity_fastapi(killall):
    key = "Requests Per Second (RPS)"
    fastapi_df = run_fastapi_benchmark()
    ls_df = run_litserve_benchmark()
    fastapi_throughput = fastapi_df[key].mean()
    ls_throughput = ls_df[key].mean()
    assert ls_throughput > fastapi_throughput
