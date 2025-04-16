import litserve as ls
from litserve import WebSocketSpec
from litserve.test_examples.websocket_spec_example import WebSocketLitAPI

if __name__ == "__main__":
    server = ls.LitServer(WebSocketLitAPI(), spec=WebSocketSpec())
    server.run()
