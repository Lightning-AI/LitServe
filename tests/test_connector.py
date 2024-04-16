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
from litserve.connector import _Connector
import pytest
import torch


@pytest.mark.parametrize(
    ("input_accelerator", "expected_accelerator"),
    [
        ("cpu", "cpu"),
        pytest.param(
            "cuda",
            "cuda",
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            None, "cuda", marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU")
        ),
        pytest.param(
            "auto",
            "cuda",
            marks=pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Only tested on Nvidia GPU"),
        ),
        pytest.param(
            "auto",
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
        pytest.param(
            None,
            "mps",
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Only tested on Apple MPS"),
        ),
    ],
)
def test_connector(input_accelerator, expected_accelerator):
    connector = _Connector(accelerator=input_accelerator)
    assert (
        connector.accelerator == expected_accelerator
    ), f"accelerator was supposed to be {expected_accelerator} but was {connector.accelerator}"

    with pytest.raises(ValueError, match="accelerator must be one of 'auto', 'cpu', 'cuda', or 'gpu'"):
        _Connector(accelerator="SUPER_CHIP")
