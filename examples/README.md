# LitServe Examples

This directory contains examples demonstrating various LitServe features.

## Batched Streaming with EOS Token Handling

**File:** `batched_streaming_with_eos.py`

This example demonstrates how to properly handle End-of-Sequence (EOS) tokens in batched streaming scenarios. When streaming responses for multiple requests in a batch, different items may finish at different times. This example shows how to signal completion for individual items to avoid sending unnecessary tokens.

### Key Concepts

- **Batched Streaming**: Process multiple requests simultaneously for better GPU utilization
- **Variable Length Sequences**: Different items in a batch may generate different numbers of tokens
- **EOS Signaling**: Return `None` from `encode_response` to signal that an item has finished
- **Efficient Token Transmission**: The server stops sending data for items that return `None`

### Running the Example

```bash
python examples/batched_streaming_with_eos.py
```

Then make requests:

```bash
# Short response (3 tokens)
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": "short"}'

# Long response (9 tokens)
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": "long"}'
```

### How It Works

In the `encode_response` method, return `None` to signal that an item has finished:

```python
def encode_response(self, output_stream, context):
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
```

When batching requests with different lengths:
- Item with 3 tokens: Receives 3 tokens, then gets finish signal
- Item with 9 tokens: Continues receiving all 9 tokens, then gets finish signal
- **Without this feature**: Item with 3 tokens would receive 6 unnecessary padding/empty tokens

### Benefits

1. **Bandwidth Efficiency**: No unnecessary token transmission after sequence completion
2. **Lower Latency**: Clients know immediately when a sequence is complete
3. **Better UX**: Cleaner streaming experience without padding tokens
4. **GPU Efficiency**: Continue batching while items finish at different times
