# server

```bash
pip install mlserver
```

## example

```python
from lib import LitAPI, LitServer

class SimpleLitAPI(LitAPI):
    def setup(self, devices):
        self.model = lambda x: x**2

    def predict(self, x):
        return self.model(x)

    def decode_request(self, request):
        return request["input"]

    def encode_response(self, output):
        return {"output": output}


api = SimpleLitAPI()
server = LitServer(api, accelerator="cuda", devices=[0, 1])
server.run(port=8888)
```

Once the server starts it generates an example client you can use like this:

```bash
python client.py
```
