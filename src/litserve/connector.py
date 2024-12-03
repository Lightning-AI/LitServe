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
import os
import platform
import subprocess
import sys
from functools import lru_cache
from typing import List, Optional, Union


class _Connector:
    def __init__(self, accelerator: str = "auto", devices: Union[List[int], int, str] = "auto"):
        accelerator = self._sanitize_accelerator(accelerator)
        if accelerator == "cpu":
            self._accelerator = "cpu"
        elif accelerator == "cuda":
            self._accelerator = "cuda"
        elif accelerator == "mps":
            self._accelerator = "mps"

        elif accelerator == "auto":
            self._accelerator = self._choose_auto_accelerator()
        elif accelerator == "gpu":
            self._accelerator = self._choose_gpu_accelerator_backend()

        if devices == "auto":
            self._devices = self._auto_device_count(self._accelerator)
        else:
            self._devices = devices

        self.check_devices_and_accelerators()

    def check_devices_and_accelerators(self):
        """Check if the devices are in a valid fomra and raise an error if they are not."""
        if self._accelerator in ["cuda", "mps"]:
            if not isinstance(self._devices, int) and not (
                isinstance(self._devices, list) and all(isinstance(device, int) for device in self._devices)
            ):
                raise ValueError(
                    "devices must be an integer or a list of integers when using 'cuda' or 'mps', "
                    f"instead got {self._devices}"
                )
        elif self._accelerator != "cpu":
            raise ValueError(f"accelerator must be one of (cuda, mps, cpu), instead got {self._accelerator}")

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

        if accelerator not in ["auto", "cpu", "mps", "cuda", "gpu", None]:
            raise ValueError("accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', or 'gpu'")

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
        if check_cuda_with_nvidia_smi() > 0:
            return "cuda"

        try:
            import torch

            if torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
                return "mps"
        except ImportError:
            return None

        return None


@lru_cache(maxsize=1)
def check_cuda_with_nvidia_smi() -> int:
    """Checks if CUDA is installed using the `nvidia-smi` command-line tool.

    Returns count of visible devices.

    """
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
        devices = [el for el in nvidia_smi_output.split("\n") if el.startswith("GPU")]
        devices = [el.split(":")[0].split()[1] for el in devices]
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            # we need check the intersection of devices and visible devices, since
            # using CUDA_VISIBLE_DEVICES=0,25 on a 4-GPU machine will yield
            # torch.cuda.device_count() == 1
            devices = [el for el in devices if el in visible_devices.split(",")]
        return len(devices)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
