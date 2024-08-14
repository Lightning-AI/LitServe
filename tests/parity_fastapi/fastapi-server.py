import base64
import io

import PIL
import torch
import torchvision
from fastapi import FastAPI, HTTPException
from jsonargparse import CLI
from pydantic import BaseModel

# Set float32 matrix multiplication precision if GPU is available and capable
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")

app = FastAPI()


class ImageData(BaseModel):
    image_data: str


class ImageClassifierAPI:
    def __init__(self, device):
        self.device = device
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.image_processing = weights.transforms()
        self.model = torchvision.models.resnet18(weights=None).eval().to(device)

    def process_image(self, image_data):
        image = base64.b64decode(image_data)
        pil_image = PIL.Image.open(io.BytesIO(image)).convert("RGB")
        processed_image = self.image_processing(pil_image)
        return processed_image.unsqueeze(0).to(self.device)  # Add batch dimension

    def predict(self, x):
        with torch.inference_mode():
            outputs = self.model(x)
            _, predictions = torch.max(outputs, 1)
        return predictions.item()


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    device = "mps"
api = ImageClassifierAPI(device)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image_data: ImageData):
    try:
        processed_image = api.process_image(image_data.image_data)
        prediction = api.predict(processed_image)
        return {"output": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    CLI(main)
