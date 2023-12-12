import os
from fastapi import FastAPI, Request
from mlserver.lit_api import LitAPI

class LitServer:
    def __init__(self, lit_api: LitAPI):
        self.app = FastAPI()
        self.lit_api = lit_api

        @self.app.on_event("startup")
        async def setup_model():
            # Setup the model during server startup
            await self.lit_api.setup()
            self.generate_client_file()

        @self.app.post("/predict/")
        async def predict(request: Request):
            # Move the data retrieval inside the LitServer
            try:
                data = await request.json()
                result = await self.lit_api.predict(request, data)
                return result
            except Exception as e:
                raise e

    def generate_client_file(self):
        src_path = "./client.py"
        dest_path = os.getcwd() + '/client.py'

        # Copy the file to the destination directory
        try:
            shutil.copy(src_path, dest_path)
            print(f"File '{src_path}' copied to '{dest_path}'")
        except Exception as e:
            print(f"Error copying file: {e}")

    def run(self, port: int):
        import uvicorn
        uvicorn.run(self.app, host="127.0.0.1", port=port, log_level="info")