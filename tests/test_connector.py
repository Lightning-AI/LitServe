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
from unittest.mock import patch

from litserve.connector import _Connector, check_cuda_with_nvidia_smi
import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU")
def test_check_cuda_with_nvidia_smi():
    assert check_cuda_with_nvidia_smi() == torch.cuda.device_count()


@pytest.mark.skipif(torch.cuda.device_count() > 0, reason="Non Nvidia GPU only")
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
        pytest.param(
            "cuda",
            "cuda",
            torch.cuda.device_count(),
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            "gpu",
            "cuda",
            torch.cuda.device_count(),
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            None,
            "cuda",
            torch.cuda.device_count(),
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            "auto",
            "cuda",
            torch.cuda.device_count(),
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            "auto",
            "mps",
            1,
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
        pytest.param(
            "gpu",
            "mps",
            1,
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
        pytest.param(
            "mps",
            "mps",
            1,
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
        pytest.param(
            None,
            "mps",
            1,
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
    ],
)
def test_connector(input_accelerator, expected_accelerator, expected_devices):
    check_cuda_with_nvidia_smi.cache_clear()
    connector = _Connector(accelerator=input_accelerator)
    assert (
        connector.accelerator == expected_accelerator
    ), f"accelerator mismatch - expected: {expected_accelerator}, actual: {connector.accelerator}"

    assert (
        connector.devices == expected_devices
    ), f"devices mismatch - expected {expected_devices}, actual: {connector.devices}"

    with pytest.raises(ValueError, match="accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', or 'gpu'"):
        _Connector(accelerator="SUPER_CHIP")


def test__sanitize_accelerator():
    assert _Connector._sanitize_accelerator(None) == "auto"
    assert _Connector._sanitize_accelerator("CPU") == "cpu"
    with pytest.raises(ValueError, match="accelerator must be one of 'auto', 'cpu', 'mps', 'cuda', or 'gpu'"):
        _Connector._sanitize_accelerator("SUPER_CHIP")
