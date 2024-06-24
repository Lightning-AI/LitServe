import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from jsonargparse import CLI
import litserve as ls


class HuggingFaceLitAPI(ls.LitAPI):
    def setup(self, device):
        model_name = "google-bert/bert-large-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # torch_dtype=torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.float16).to(
            device
        )

    def decode_request(self, request):
        return request["text"]

    def batch(self, inputs):
        print(len(inputs))
        return self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

    @torch.inference_mode
    def predict(self, inputs):
        inputs = inputs.to(self.device)
        logits = self.model(**inputs).logits
        predicted_class_ids = logits.argmax(1)
        return predicted_class_ids

    def unbatch(self, outputs):
        return outputs.tolist()

    def encode_response(self, output):
        return {"label_idx": output}


def main(
    batch_size: int = 8,
    batch_timeout: float = 0.01,
):
    print(batch_size, batch_timeout)
    api = HuggingFaceLitAPI()
    server = ls.LitServer(
        api,
        max_batch_size=batch_size,
        workers_per_device=1,
        batch_timeout=batch_timeout,
        timeout=200,
    )
    server.run()


if __name__ == "__main__":
    CLI(main)
