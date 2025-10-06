#!/usr/bin/env python
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
"""
Example: Batched Streaming with EOS Token Handling

This example demonstrates how to properly handle End-of-Sequence (EOS) tokens
in batched streaming scenarios. When streaming responses for multiple requests
in a batch, different items may finish at different times. By returning None
in encode_response, you can signal that a particular item has finished, preventing
unnecessary tokens from being sent to the client.

Key Concepts:
1. Batched streaming processes multiple requests simultaneously
2. Items in a batch may generate different numbers of tokens
3. Return None from encode_response to signal EOS for that item
4. The server will stop sending data for items that return None

Usage:
    python examples/batched_streaming_with_eos.py

Then make requests:
    curl -X POST http://127.0.0.1:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "short"}'
    
    curl -X POST http://127.0.0.1:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "long"}'
"""

import litserve as ls


class BatchedStreamingWithEOSAPI(ls.LitAPI):
    """
    Example API that demonstrates proper EOS token handling in batched streaming.
    
    This simulates an LLM that generates different length sequences for different
    prompts. Items that finish early return None to signal they're done, preventing
    unnecessary token transmission.
    """
    
    def setup(self, device):
        """Initialize the model (simulated in this example)."""
        self.model = None
        
        # Simulated responses for different prompt types
        self.responses = {
            "short": ["Hello", "world", "!"],
            "medium": ["This", "is", "a", "test", "response"],
            "long": ["Here", "is", "a", "much", "longer", "response", "with", "many", "tokens"],
        }
    
    def decode_request(self, request):
        """Extract the prompt from the request."""
        return request["prompt"]
    
    def batch(self, inputs):
        """Pass through the inputs as a batch."""
        return inputs
    
    def predict(self, prompts, context):
        """
        Generate tokens for a batch of prompts.
        
        Different prompts generate different numbers of tokens. The predict method
        yields batches of tokens where each element corresponds to one item in the
        input batch.
        """
        n = len(prompts)
        
        # Get the responses for each prompt
        sequences = []
        for prompt in prompts:
            if prompt in self.responses:
                sequences.append(self.responses[prompt])
            else:
                # Default response
                sequences.append(["Default", "response"])
        
        # Find the maximum sequence length
        max_len = max(len(seq) for seq in sequences)
        
        # Yield tokens for each position
        # When a sequence is done, yield None for that position
        for i in range(max_len):
            batch_output = []
            for seq in sequences:
                if i < len(seq):
                    batch_output.append(seq[i])
                else:
                    # This item has finished - mark as None
                    batch_output.append(None)
            yield batch_output
    
    def unbatch(self, output):
        """Pass through the batched output."""
        return output
    
    def encode_response(self, output_stream, context):
        """
        Encode the streaming output for each item in the batch.
        
        IMPORTANT: Return None for an item to signal that it has finished (EOS).
        The server will stop sending data for that item and send the finish signal.
        
        This prevents sending unnecessary padding tokens or empty responses after
        an item has completed its generation.
        """
        for outputs in output_stream:
            encoded_batch = []
            for output in outputs:
                if output is None:
                    # Signal EOS - this item is done
                    encoded_batch.append(None)
                else:
                    # Encode the token
                    encoded_batch.append({"role": "assistant", "content": output})
            yield encoded_batch


def main():
    """Run the server with batched streaming and EOS handling."""
    api = BatchedStreamingWithEOSAPI(
        stream=True,
        max_batch_size=4,
        batch_timeout=0.01,  # Small timeout for batching
    )
    
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000)


if __name__ == "__main__":
    main()
