import litserve as ls

if __name__ == "__main__":
    api = ls.examples.SimpleLitAPI()
    server = ls.LitServer(api)
    server.run(port=8000)
