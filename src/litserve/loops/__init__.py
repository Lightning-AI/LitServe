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
import multiprocessing as mp

from litserve.loops.base import (
    _BaseLoop,
    run_batched_loop,
    run_batched_streaming_loop,
    run_single_loop,
    run_streaming_loop,
)
from litserve.loops.continuous_batching_loop import ContinuousBatchingLoop, Output
from litserve.loops.loops import (
    BatchedLoop,
    BatchedStreamingLoop,
    DefaultLoop,
    LitLoop,
    SingleLoop,
    StreamingLoop,
    get_default_loop,
    inference_worker,
)

mp.allow_connection_pickling()

__all__ = [
    "_BaseLoop",
    "DefaultLoop",
    "SingleLoop",
    "StreamingLoop",
    "BatchedLoop",
    "BatchedStreamingLoop",
    "ContinuousBatchingLoop",
    "LitLoop",
    "run_batched_loop",
    "run_streaming_loop",
    "run_batched_streaming_loop",
    "run_single_loop",
    "get_default_loop",
    "inference_worker",
    "Output",
]
