import litserve as ls

if __name__ == "__main__":
    api = ls.examples.SimpleBatchedAPI()
    server = ls.LitServer(api, max_batch_size=4, batch_timeout=0.05)
    server.run(port=8000)
