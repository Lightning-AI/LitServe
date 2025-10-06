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
"""Tests for batched streaming with EOS token handling."""

import pytest

import litserve as ls


class BatchedStreamingEOSTestAPI(ls.LitAPI):
    """Test API that generates different length sequences."""
    
    def setup(self, device):
        self.model = None
    
    def batch(self, inputs):
        return inputs
    
    def predict(self, inputs, context):
        """
        Generate sequences of different lengths.
        Item 0: 3 tokens
        Item 1: 5 tokens
        """
        n = len(inputs)
        sequences = [
            ["token1", "token2", "token3"],
            ["word1", "word2", "word3", "word4", "word5"],
        ]
        
        max_len = max(len(seq) for seq in sequences[:n])
        
        for i in range(max_len):
            batch_output = []
            for j in range(n):
                if j < len(sequences) and i < len(sequences[j]):
                    batch_output.append(sequences[j][i])
                else:
                    batch_output.append(None)
            yield batch_output
    
    def unbatch(self, output):
        return output
    
    def encode_response(self, output_stream, context):
        """Encode responses, returning None for finished items."""
        for outputs in output_stream:
            encoded_batch = []
            for output in outputs:
                if output is None:
                    encoded_batch.append(None)
                else:
                    encoded_batch.append({"content": output})
            yield encoded_batch


@pytest.mark.asyncio
async def test_batched_streaming_with_eos():
    """Test that items in a batch can finish at different times."""
    api = BatchedStreamingEOSTestAPI()
    api.setup(None)
    
    # Simulate batch of 2 requests
    inputs = ["request1", "request2"]
    context = [{}, {}]
    
    batched_inputs = api.batch(inputs)
    output_stream = api.predict(batched_inputs, context)
    unbatched_stream = api.unbatch(output_stream)
    encoded_stream = list(api.encode_response(unbatched_stream, context))
    
    # Expected behavior:
    # Step 0: Both items produce tokens
    # Step 1: Both items produce tokens
    # Step 2: Both items produce tokens
    # Step 3: Item 0 is None (finished), Item 1 produces token
    # Step 4: Item 0 is None (finished), Item 1 produces token
    
    assert len(encoded_stream) == 5, "Should have 5 steps"
    
    # Step 0
    assert encoded_stream[0][0] == {"content": "token1"}
    assert encoded_stream[0][1] == {"content": "word1"}
    
    # Step 1
    assert encoded_stream[1][0] == {"content": "token2"}
    assert encoded_stream[1][1] == {"content": "word2"}
    
    # Step 2
    assert encoded_stream[2][0] == {"content": "token3"}
    assert encoded_stream[2][1] == {"content": "word3"}
    
    # Step 3 - Item 0 finished
    assert encoded_stream[3][0] is None, "Item 0 should be finished (None)"
    assert encoded_stream[3][1] == {"content": "word4"}
    
    # Step 4 - Item 0 still finished
    assert encoded_stream[4][0] is None, "Item 0 should still be finished (None)"
    assert encoded_stream[4][1] == {"content": "word5"}


@pytest.mark.asyncio
async def test_batched_streaming_all_finish_together():
    """Test that when all items finish at the same time, all return None."""
    
    class SameLengthAPI(ls.LitAPI):
        def setup(self, device):
            self.model = None
        
        def batch(self, inputs):
            return inputs
        
        def predict(self, inputs, context):
            """All items generate same length sequences."""
            n = len(inputs)
            for i in range(3):
                yield [f"token{i}"] * n
        
        def unbatch(self, output):
            return output
        
        def encode_response(self, output_stream, context):
            for outputs in output_stream:
                yield [{"content": output} for output in outputs]
    
    api = SameLengthAPI()
    api.setup(None)
    
    inputs = ["req1", "req2"]
    context = [{}, {}]
    
    batched_inputs = api.batch(inputs)
    output_stream = api.predict(batched_inputs, context)
    unbatched_stream = api.unbatch(output_stream)
    encoded_stream = list(api.encode_response(unbatched_stream, context))
    
    # All items should finish at the same time
    assert len(encoded_stream) == 3
    for step in encoded_stream:
        assert all(item["content"].startswith("token") for item in step)


@pytest.mark.asyncio
async def test_batched_streaming_single_item_finishes_first():
    """Test that the first item can finish before others."""
    
    class FirstFinishesAPI(ls.LitAPI):
        def setup(self, device):
            self.model = None
        
        def batch(self, inputs):
            return inputs
        
        def predict(self, inputs, context):
            """First item finishes quickly."""
            sequences = [
                ["quick"],
                ["slow", "generation", "continues"],
            ]
            
            for i in range(3):
                batch = []
                for j in range(len(inputs)):
                    if i < len(sequences[j]):
                        batch.append(sequences[j][i])
                    else:
                        batch.append(None)
                yield batch
        
        def unbatch(self, output):
            return output
        
        def encode_response(self, output_stream, context):
            for outputs in output_stream:
                encoded = []
                for output in outputs:
                    if output is None:
                        encoded.append(None)
                    else:
                        encoded.append({"content": output})
                yield encoded
    
    api = FirstFinishesAPI()
    api.setup(None)
    
    inputs = ["req1", "req2"]
    context = [{}, {}]
    
    batched_inputs = api.batch(inputs)
    output_stream = api.predict(batched_inputs, context)
    unbatched_stream = api.unbatch(output_stream)
    encoded_stream = list(api.encode_response(unbatched_stream, context))
    
    # Item 0 should finish after first token
    assert encoded_stream[0][0] == {"content": "quick"}
    assert encoded_stream[1][0] is None  # Item 0 finished
    assert encoded_stream[2][0] is None  # Item 0 still finished
    
    # Item 1 continues
    assert encoded_stream[0][1] == {"content": "slow"}
    assert encoded_stream[1][1] == {"content": "generation"}
    assert encoded_stream[2][1] == {"content": "continues"}
