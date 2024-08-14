import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor

import PIL
import torch
import torchvision

import litserve as ls

# Set float32 matrix multiplication precision if GPU is available and capable
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")


class ImageClassifierAPI(ls.LitAPI):
    def setup(self, device):
        print(device)
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.image_processing = weights.transforms()
        self.model = torchvision.models.resnet18(weights=None).eval().to(device)
        self.pool = ThreadPoolExecutor(os.cpu_count())

    def decode_request(self, request):
        return request["image_data"]

    def batch(self, image_data_list):
        def process_image(image_data):
            image = base64.b64decode(image_data)
            pil_image = PIL.Image.open(io.BytesIO(image)).convert("RGB")
            return self.image_processing(pil_image)

        inputs = list(self.pool.map(process_image, image_data_list))
        return torch.stack(inputs).to(self.device)

    def predict(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
            _, predictions = torch.max(outputs, 1)
        return predictions

    def unbatch(self, outputs):
        return outputs.tolist()

    def encode_response(self, output):
        return {"output": output}


def main(batch_size: int = 8, workers_per_device: int = 1):
    print(locals())
    api = ImageClassifierAPI()
    server = ls.LitServer(
        api,
        max_batch_size=batch_size,
        batch_timeout=0.01,
        timeout=10,
        workers_per_device=workers_per_device,
    )
    server.run(port=8000)


if __name__ == "__main__":
    conf = {
        "cuda": {"batch_size": 8, "workers_per_device": 1},
        "cpu": {"batch_size": 4, "workers_per_device": 1},
        "mps": {"batch_size": 4, "workers_per_device": 1},
    }
    device = "cpu" if torch.cuda.is_available() else "cuda"
    device = "mps" if torch.backends.mps.is_available() else device
    main(**conf[device])
