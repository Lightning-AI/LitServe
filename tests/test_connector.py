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
import platform
from unittest.mock import MagicMock, patch

import pytest

from litserve.connector import _Connector, check_cuda_with_nvidia_smi

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    import jax.extend.backend as jax_backend

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def jax_mps_check():
    if not JAX_AVAILABLE:
        return False
    # Check if JAX is configured for MPS
    return (
        any(d.device_kind == "mps" for d in jax.devices())
        and (jax_backend.get_backend().platform == "mps")
        and platform.processor() in ("arm", "arm64")
    )


def torch_mps_check():
    if not TORCH_AVAILABLE:
        return False
    return torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64")


@pytest.mark.skipif(
    not TORCH_AVAILABLE or torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU with PyTorch"
)
def test_check_cuda_with_nvidia_smi():
    assert check_cuda_with_nvidia_smi() == torch.cuda.device_count()


@pytest.mark.skipif(TORCH_AVAILABLE and torch.cuda.device_count() > 0, reason="Non-Nvidia GPU only")
@patch(
    "litserve.connector.subprocess.check_output",
    return_value=b"GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-rb438fre-0ar-9702-de35-ref4rjn34omk3 )",
)
def test_check_cuda_with_nvidia_smi_mock_gpu(mock_subprocess):
    check_cuda_with_nvidia_smi.cache_clear()
    assert check_cuda_with_nvidia_smi() == 1
    check_cuda_with_nvidia_smi.cache_clear()


@pytest.mark.parametrize(
    ("input_accelerator", "expected_accelerator", "expected_devices"),
    [
        ("cpu", "cpu", 1),
        # --- PyTorch CUDA Tests ---
        pytest.param(
            "cuda",
            "cuda",
            lambda: torch.cuda.device_count() if TORCH_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not TORCH_AVAILABLE or torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU with PyTorch"
            ),
            id="torch_cuda_explicit",
        ),
        pytest.param(
            "gpu",
            "cuda",
            lambda: torch.cuda.device_count() if TORCH_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not TORCH_AVAILABLE or torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU with PyTorch"
            ),
            id="torch_gpu_auto_cuda",
        ),
        pytest.param(
            "auto",
            "cuda",
            lambda: torch.cuda.device_count() if TORCH_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not TORCH_AVAILABLE or torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU with PyTorch"
            ),
            id="torch_auto_cuda",
        ),
        # --- PyTorch MPS Tests ---
        pytest.param(
            "mps",
            "mps",
            1,
            marks=pytest.mark.skipif(not torch_mps_check(), reason="Only tested on Apple MPS with PyTorch"),
            id="torch_mps_check_explicit",
        ),
        pytest.param(
            "auto",
            "mps",
            1,
            marks=pytest.mark.skipif(not torch_mps_check(), reason="Only tested on Apple MPS with PyTorch"),
            id="torch_auto_mps",
        ),
        pytest.param(
            "gpu",
            "mps",
            1,
            marks=pytest.mark.skipif(not torch_mps_check(), reason="Only tested on Apple MPS with PyTorch"),
            id="torch_gpu_auto_mps",
        ),
        # --- JAX CUDA Tests  ---
        pytest.param(
            "jax",
            "jax",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not JAX_AVAILABLE
                or not any(d.device_kind == "gpu" for d in jax.devices())
                or check_cuda_with_nvidia_smi() == 0,
                reason="Only tested on Nvidia GPU with JAX",
            ),
            id="jax_cuda_explicit",
        ),
        pytest.param(
            "auto",
            "cuda",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not JAX_AVAILABLE
                or not any(d.device_kind == "gpu" for d in jax.devices())
                or check_cuda_with_nvidia_smi() == 0,
                reason="Only tested on Nvidia GPU with JAX",
            ),
            id="jax_auto_cuda",
        ),
        pytest.param(
            "gpu",
            "cuda",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not JAX_AVAILABLE
                or not any(d.device_kind == "gpu" for d in jax.devices())
                or check_cuda_with_nvidia_smi() == 0,
                reason="Only tested on Nvidia GPU with JAX",
            ),
            id="jax_gpu_auto_cuda",
        ),
        # --- JAX MPS Tests ---
        pytest.param(
            "jax",
            "jax",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(not jax_mps_check(), reason="Only tested on Apple MPS with JAX"),
            id="jax_mps_check_explicit",
        ),
        pytest.param(
            "auto",
            "mps",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(not jax_mps_check(), reason="Only tested on Apple MPS with JAX"),
            id="jax_auto_mps",
        ),
        pytest.param(
            "gpu",
            "mps",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(not jax_mps_check(), reason="Only tested on Apple MPS with JAX"),
            id="jax_gpu_auto_mps",
        ),
        # --- None (auto) tests, consider both Torch and JAX scenarios ---
        pytest.param(
            None,
            "cuda",
            lambda: torch.cuda.device_count() if TORCH_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                not TORCH_AVAILABLE or torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU with PyTorch"
            ),
            id="none_auto_cuda_torch",
        ),
        pytest.param(
            None,
            "mps",
            1,
            marks=pytest.mark.skipif(not torch_mps_check(), reason="Only tested on Apple MPS with PyTorch"),
            id="none_auto_mps_torch",
        ),
        pytest.param(
            None,
            "mps",
            lambda: jax.device_count() if JAX_AVAILABLE else 0,
            marks=pytest.mark.skipif(
                TORCH_AVAILABLE and torch_mps_check(),
                reason="PyTorch MPS takes precedence for 'auto' if available. Test JAX auto-mps only if PyTorch MPS is absent.",
            ),
            id="none_auto_mps_jax_fallback",
        ),
    ],
)
def test_connector(input_accelerator, expected_accelerator, expected_devices):
    check_cuda_with_nvidia_smi.cache_clear()

    if callable(expected_devices):
        expected_devices = expected_devices()

    connector = _Connector(accelerator=input_accelerator)
    assert connector.accelerator == expected_accelerator, (
        f"accelerator mismatch - expected: {expected_accelerator}, actual: {connector.accelerator}"
    )

    assert connector.devices == expected_devices, (
        f"devices mismatch - expected {expected_devices}, actual: {connector.devices}"
    )

    with pytest.raises(ValueError, match="accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', 'gpu', or 'jax'"):
        _Connector(accelerator="SUPER_CHIP")


def test__sanitize_accelerator():
    assert _Connector._sanitize_accelerator(None) == "auto"
    assert _Connector._sanitize_accelerator("CPU") == "cpu"
    assert _Connector._sanitize_accelerator("JAX") == "jax"
    with pytest.raises(ValueError, match="accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', 'gpu', or 'jax'"):
        _Connector._sanitize_accelerator("SUPER_CHIP")


# --- Mocking tests for JAX scenarios ---


@pytest.mark.skipif(JAX_AVAILABLE, reason="Only run mocking tests if JAX is not actually installed")
@patch("litserve.connector.jax")
@patch("litserve.connector.jax_backend")
@patch("litserve.connector.check_cuda_with_nvidia_smi", return_value=0)  # No CUDA
def test_connector_mock_jax_mps_check(mock_cuda_smi, mock_jax_backend, mock_jax):
    # Mock JAX to report MPS platform and an MPS device
    mock_jax_backend.get_backend.return_value.platform = "mps"
    mock_jax.devices.return_value = [MagicMock(device_kind="mps")]
    mock_jax.device_count.return_value = 1

    # Mock platform.processor for Apple Silicon
    with patch("platform.processor", return_value="arm64"):
        check_cuda_with_nvidia_smi.cache_clear()
        connector = _Connector(accelerator="auto")
        assert connector.accelerator == "mps"
        assert connector.devices == 1

        check_cuda_with_nvidia_smi.cache_clear()
        connector = _Connector(accelerator="jax")
        assert connector.accelerator == "jax"
        assert connector.devices == 1

        check_cuda_with_nvidia_smi.cache_clear()
        connector = _Connector(accelerator="gpu")
        assert connector.accelerator == "mps"
        assert connector.devices == 1


@pytest.mark.skipif(JAX_AVAILABLE, reason="Only run mocking tests if JAX is not actually installed")
@patch("litserve.connector.jax")
@patch("litserve.connector.jax_backend")
@patch("litserve.connector.check_cuda_with_nvidia_smi", return_value=1)
def test_connector_mock_jax_cuda(mock_cuda_smi, mock_jax_backend, mock_jax):
    # Mock JAX to report CUDA platform and a GPU device
    mock_jax_backend.get_backend.return_value.platform = "cuda"  # or "gpu"
    mock_jax.devices.return_value = [MagicMock(device_kind="gpu")]
    mock_jax.device_count.return_value = 1

    check_cuda_with_nvidia_smi.cache_clear()
    connector = _Connector(accelerator="auto")
    assert connector.accelerator == "cuda"
    assert connector.devices == 1

    check_cuda_with_nvidia_smi.cache_clear()
    connector = _Connector(accelerator="jax")
    assert connector.accelerator == "jax"
    assert connector.devices == 1

    check_cuda_with_nvidia_smi.cache_clear()
    connector = _Connector(accelerator="gpu")
    assert connector.accelerator == "cuda"
    assert connector.devices == 1


@pytest.mark.skipif(JAX_AVAILABLE, reason="Only run mocking tests if JAX is not actually installed")
@patch("litserve.connector.jax", None)  # Simulate JAX not installed
@patch("litserve.connector.jax_backend", None)  # Simulate JAX backend extension not available
@patch("litserve.connector.torch", None)  # Simulate Torch not installed
@patch("litserve.connector.check_cuda_with_nvidia_smi", return_value=0)  # No CUDA
def test_connector_no_frameworks(mock_cuda_smi):
    check_cuda_with_nvidia_smi.cache_clear()
    connector = _Connector(accelerator="auto")
    assert connector.accelerator == "cpu"
    assert connector.devices == 1

    connector = _Connector(accelerator="gpu")
    assert connector.accelerator is None
    assert connector.devices == 1

    connector = _Connector(accelerator="jax")
    assert connector.accelerator == "jax"
    assert connector.devices == 1
