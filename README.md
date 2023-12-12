# server

```bash
pip install mlserver
```

## example

```python
from lib import LitAPI, LitServer
from fastapi import Request

class DefaultLitAPI(LitAPI):
    async def setup(self):
        self.model = lambda x: x**2

    async def predict(self, request: Request, data: dict):
        x = float(data.get("input", 0.0))
        result = self.model(x)
        return {"result": result}


server = LitServer(DefaultLitAPI())
server.run(port=8888)
```