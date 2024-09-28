# from litserve.server import LitServer
# from litserve.api import LitAPI
from src.litserve.server import LitServer
from src.litserve.server import LitAPI

class SimpleLitAPI(LitAPI):
    def setup(self, device):
        # setup is called once at startup. Build a compound AI system (1+ models), connect DBs, load data, etc...
        self.model1 = lambda x: x**2
        self.model2 = lambda x: x**3

    def decode_request(self, request):
        # Convert the request payload to model input.
        return request["input"] 

    def predict(self, x):
        # Easily build compound systems. Run inference and return the output.
        squared = self.model1(x)
        cubed = self.model2(x)
        output = squared + cubed
        return {"output": output}

    def encode_response(self, output):
        # Convert the model output to a response payload.
        return {"output": output} 

def main():
    server = LitServer(SimpleLitAPI(), accelerator="auto", max_batch_size=1)
    
    # TEST: testing the server can run and and receive requests from client.py on non-8000 port
    server.run(port=8080)
