import base64
import io
import os

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import litserve as ls
from litserve.schema.image import ImageInput, ImageOutput
from litserve.utils import wrap_litserve_start


class ImageAPI(ls.LitAPI):
    def setup(self, device):
        self.model = lambda x: np.array(x) * 2

    def decode_request(self, request: ImageInput):
        return request.get_image()

    def predict(self, x):
        return self.model(x)

    def encode_response(self, numpy_image) -> ImageOutput:
        output = Image.fromarray(np.uint8(numpy_image)).convert("RGB")
        return ImageOutput(image=output)


def test_image_input_output(tmpdir):
    path = os.path.join(tmpdir, "test.png")
    server = ls.LitServer(ImageAPI(), accelerator="cpu", devices=1, workers_per_device=1)
    with wrap_litserve_start(server) as server, TestClient(server.app) as client:
        Image.new("RGB", (32, 32)).save(path)
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        response = client.post("/predict", json={"image_data": encoded_string})

        assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
        image_data = response.json()["image"]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        assert image.size == (32, 32), f"Unexpected image size: {image.size}"
