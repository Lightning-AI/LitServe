"""Serve FunASR SenseVoiceSmall with LitServe.

Install the optional runtime dependency before running this example:

    pip install funasr

Start the server:

    python examples/funasr_sensevoice.py

Send either JSON:

    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"audio_path": "sample.wav"}'

or multipart form data:

    curl -X POST http://127.0.0.1:8000/predict \
      -F "file=@sample.wav"

"""

import tempfile
from pathlib import Path

from fastapi import Request
import litserve as ls


class FunASRSenseVoiceAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        from funasr import AutoModel

        self.model = AutoModel(model="iic/SenseVoiceSmall", device=device)

    def decode_request(self, request: Request) -> Path:
        if isinstance(request, dict) and "audio_path" in request:
            return Path(request["audio_path"])

        if isinstance(request, dict) and "file" in request:
            upload = request["file"]
            suffix = Path(getattr(upload, "filename", "audio.wav")).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.file.read())
                return Path(tmp.name)

        raise ValueError("Send JSON with audio_path or multipart form data with file.")

    def predict(self, audio_path: Path) -> dict[str, str]:
        result = self.model.generate(input=str(audio_path))
        text = result[0].get("text", "") if result else ""
        return {"text": text}

    def encode_response(self, output: dict[str, str]) -> dict[str, str]:
        return output


if __name__ == "__main__":
    server = ls.LitServer(FunASRSenseVoiceAPI(), accelerator="auto")
    server.run(port=8000)
