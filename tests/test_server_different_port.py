import requests
import multiprocessing as mp

# NOTE: HAD TO ADD THIS TO RUN LOCALLY W/O ERROR
if __name__ == "__main__":
    mp.set_start_method('fork')  # Try using 'spawn' if 'fork' doesn't work


from litserve.server import LitServer
from litserve.api import LitAPI

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

def test_server_different_port():
    server = LitServer(SimpleLitAPI(), accelerator="auto", max_batch_size=1)
    
    # TEST: testing the server can run and and receive requests from client.py on non-8000 port
    non_default_port_no = 8080
    server.run(port=non_default_port_no)

    response = requests.post(f"http://127.0.0.1:{non_default_port_no}/predict", json={"input": 4.0})
    print(response)
    assert response.status_code == 200


# Couldn't get pytest to run this, so calling the method and running in the terminal
test_server_different_port()


