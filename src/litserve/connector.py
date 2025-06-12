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
import logging
import os
import platform
import subprocess
from functools import lru_cache
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class _Connector:
    def __init__(self, accelerator: str = "auto", devices: Union[List[int], int, str] = "auto"):
        accelerator = self._sanitize_accelerator(accelerator)
        if accelerator in ("cpu", "cuda", "mps", "jax"):
            self._accelerator = accelerator
        elif accelerator == "auto":
            self._accelerator = self._choose_auto_accelerator()
        elif accelerator == "gpu":
            self._accelerator = self._choose_gpu_accelerator_backend()

        if devices == "auto":
            self._devices = self._accelerator_device_count()
        else:
            self._devices = devices

        self.check_devices_and_accelerators()

    def check_devices_and_accelerators(self):
        """Check if the devices are in a valid fomra and raise an error if they are not."""
        if self._accelerator in ("cuda", "mps", "jax"):
            if not isinstance(self._devices, int) and not (
                isinstance(self._devices, list) and all(isinstance(device, int) for device in self._devices)
            ):
                raise ValueError(
                    "devices must be an integer or a list of integers when using 'cuda', 'mps', or 'jax', "
                    f"instead got {self._devices}"
                )
        elif self._accelerator != "cpu":
            # Updated error message to include 'jax'
            raise ValueError(f"accelerator must be one of (cuda, mps, cpu, jax), instead got {self._accelerator}")

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

        if accelerator not in ["auto", "cpu", "mps", "cuda", "gpu", "jax", None]:
            raise ValueError(
                f"accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', 'gpu', or 'jax'. Found: {accelerator}"
            )

        if accelerator is None:
            return "auto"
        return accelerator

    def _choose_auto_accelerator(self):
        """Determines the appropriate accelerator for 'auto' mode, with PyTorch preference then JAX."""
        torch_backend = self._choose_gpu_accelerator_torch()
        if torch_backend:
            logger.info(f"Auto-selected PyTorch GPU accelerator: {torch_backend}")
            return torch_backend

        jax_backend = self._choose_gpu_accelerator_jax()
        if jax_backend:
            logger.info(f"Auto-selected JAX GPU accelerator: {jax_backend}")
            return jax_backend

        logger.info("No GPU accelerator detected for 'auto' mode. Defaulting to 'cpu'.")
        return "cpu"

    def _accelerator_device_count(self) -> int:
        if self._accelerator == "cuda":
            return check_cuda_with_nvidia_smi()
        if self._accelerator == "mps":
            return 1  # MPS typically only has 1 GPU
        if self._accelerator == "jax":
            try:
                import jax

                return jax.device_count()
            except ImportError:
                logger.warning("JAX not installed, cannot determine JAX device count. Defaulting to 1.")
                return 1
        return 1

    @staticmethod
    def _choose_gpu_accelerator_torch():
        """Checks for PyTorch GPU accelerator backend (CUDA or MPS)."""
        if check_cuda_with_nvidia_smi() > 0:
            return "cuda"

        try:
            import torch

            if torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
                return "mps"
        except ImportError:
            logger.debug("PyTorch not installed, skipping PyTorch GPU accelerator check.")
        return None

    @staticmethod
    def _choose_gpu_accelerator_jax():
        """Checks for JAX GPU accelerator backend (CUDA or MPS)."""
        try:
            import jax
            from jax.extend import backend as jax_backend

            # JAX with CUDA
            if jax_backend.get_backend().platform == "gpu" or jax_backend.get_backend().platform == "cuda":
                if check_cuda_with_nvidia_smi() > 0:
                    return "cuda"

            # JAX with MPS
            if platform.processor() in ("arm", "arm64"):
                if jax_backend.get_backend().platform == "mps":
                    return "mps"

        except ImportError:
            logger.debug("JAX not installed, skipping JAX GPU accelerator check.")
        except Exception as e:
            logger.debug(f"Error during JAX GPU accelerator check: {e}")

        return None

    @staticmethod
    def _choose_gpu_accelerator_backend():
        """Determines the appropriate GPU accelerator backend when `accelerator='gpu'` is explicitly set.

        Prioritizes PyTorch then JAX.

        """
        torch_backend = _Connector._choose_gpu_accelerator_torch()
        if torch_backend:
            logger.info(f"Explicit 'gpu' accelerator selected PyTorch backend: {torch_backend}")
            return torch_backend

        jax_backend = _Connector._choose_gpu_accelerator_jax()
        if jax_backend:
            logger.info(f"Explicit 'gpu' accelerator selected JAX backend: {jax_backend}")
            return jax_backend

        logger.info("No specific GPU accelerator detected for explicit 'gpu' selection.")
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
