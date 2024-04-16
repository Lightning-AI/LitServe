# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from typing import Optional
import platform


class _Connector:
    def __init__(self, accelerator: Optional[str] = "auto"):
        accelerator = self._sanitize_accelerator(accelerator)
        if accelerator == "cpu":
            self._accelerator = "cpu"
        elif accelerator == "cuda":
            self._accelerator = "cuda"

        elif accelerator == "auto":
            self._accelerator = self._choose_auto_accelerator()
        elif accelerator == "gpu":
            self._accelerator = self._choose_gpu_accelerator_backend()

    @property
    def accelerator(self):
        return self._accelerator

    @staticmethod
    def _sanitize_accelerator(accelerator: Optional[str]):
        if isinstance(accelerator, str):
            accelerator = accelerator.lower()

        if accelerator not in ["auto", "cpu", "cuda", "gpu", None]:
            raise ValueError("accelerator must be one of 'auto', 'cpu', 'cuda', or 'gpu'")

        if accelerator is None:
            return "auto"
        return accelerator

    def _choose_auto_accelerator(self):
        if "torch" in sys.modules:
            try:
                return self._choose_gpu_accelerator_backend()
            except RuntimeError:
                print("No GPU accelerator found, will default to cpu")
        return "cpu"

    @staticmethod
    def _choose_gpu_accelerator_backend():
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
            return "mps"
        raise RuntimeError("No supported gpu backend found!")
