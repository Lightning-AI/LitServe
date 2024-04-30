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
from typing import List, Optional, Union
import platform
import subprocess
from functools import lru_cache


class _Connector:
    def __init__(self, accelerator: str = "auto", devices: Union[List[int], int, str] = "auto"):
        accelerator = self._sanitize_accelerator(accelerator)
        if accelerator == "cpu":
            self._accelerator = "cpu"
        elif accelerator == "cuda":
            self._accelerator = "cuda"

        elif accelerator == "auto":
            self._accelerator = self._choose_auto_accelerator()
        elif accelerator == "gpu":
            self._accelerator = self._choose_gpu_accelerator_backend()

        if devices == "auto":
            self._devices = self._auto_device_count(self._accelerator)
        else:
            self._devices = devices

    @property
    def accelerator(self):
        return self._accelerator

    @property
    def devices(self):
        return self._devices

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
        gpu_backend = self._choose_gpu_accelerator_backend()
        if "torch" in sys.modules and gpu_backend:
            return gpu_backend
        return "cpu"

    def _auto_device_count(self, accelerator) -> int:
        if accelerator == "cuda":
            return check_cuda_with_nvidia_smi()
        return 1

    @staticmethod
    def _choose_gpu_accelerator_backend():
        import torch

        if check_cuda_with_nvidia_smi() > 0:
            return "cuda"
        if torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
            return "mps"
        return None


@lru_cache(maxsize=1)
def check_cuda_with_nvidia_smi():
    """Checks if CUDA is installed using the `nvidia-smi` command-line tool."""
    command = ["nvidia-smi", "--list-gpus"]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return len(result.stdout.splitlines())
        print("Error:", result.stderr)
        return 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
