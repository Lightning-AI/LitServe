"""A BERT-Large text classification server with batching to be used for benchmarking."""

import torch
from jsonargparse import CLI
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertConfig

import litserve as ls

# Set float32 matrix multiplication precision if GPU is available and capable
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    torch.set_float32_matmul_precision("high")

# set dtype to bfloat16 if CUDA is available
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class HuggingFaceLitAPI(ls.LitAPI):
    def setup(self, device):
        print(device)
        model_name = "google-bert/bert-large-uncased"
        config = BertConfig.from_pretrained(pretrained_model_name_or_path=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_config(config, torch_dtype=dtype)
        self.model.to(device)

    def decode_request(self, request: dict):
        return request["text"]

    def batch(self, inputs):
        return self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

    def predict(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            return torch.argmax(logits, dim=1)

    def unbatch(self, outputs):
        return outputs.tolist()

    def encode_response(self, output):
        return {"label_idx": output}


def main(
    batch_size: int = 10,
    batch_timeout: float = 0.01,
    devices: int = 2,
    workers_per_device=2,
):
    print(locals())
    api = HuggingFaceLitAPI(
        max_batch_size=batch_size,
        batch_timeout=batch_timeout,
    )
    server = ls.LitServer(
        api,
        workers_per_device=workers_per_device,
        accelerator="auto",
        devices=1,
        timeout=200,
        fast_queue=True,
    )
    server.run(log_level="warning", num_api_servers=4, generate_client_file=False)


if __name__ == "__main__":
    CLI(main)
