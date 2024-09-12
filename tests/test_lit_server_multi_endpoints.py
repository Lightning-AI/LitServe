from litserve.server import LitServer


async def test_lit_server_with_multi_endpoints(simple_litapi):
    server1 = LitServer(simple_litapi, api_path="/predict-1", timeout=10)
    server2 = LitServer(simple_litapi, api_path="/predict-2", timeout=10)
    servers = [server1, server2]
    # TODO: update test to use run_all

    # Run the servers in a separate thread
    port = 8000
    # server_thread = threading.Thread(
    #     target=run_all, args=(servers,), kwargs={"port": port, "num_api_servers": 2, "log_level": "debug"}
    # )
    # server_thread.start()

    # async with AsyncClient(base_url=f"http://localhost:{port}") as client:
    #     # Test server1 endpoint
    #     response1 = await client.post("/predict-1", json={"input": 1})
    #     assert response1.status_code == 200
    #     assert response1.json() == {"output": 1}

    #     # Test server2 endpoint
    #     response2 = await client.post("/predict-2", json={"input": 2})
    #     assert response2.status_code == 200
    #     assert response2.json() == {"output": 2}
