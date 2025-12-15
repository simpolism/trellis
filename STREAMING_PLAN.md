# Streaming Generation for Trellis: Technical Analysis

## Current Implementation

The engine uses HuggingFace's `model.generate()` with `num_return_sequences=4` to generate all options in a single batched call:

```python
# engine/unsloth_engine.py lines 130-139
with torch.no_grad():
    sequences = self.model.generate(
        inputs,
        max_new_tokens=self.config.max_new_tokens,
        temperature=self.config.temperature,
        min_p=self.config.min_p,
        num_return_sequences=self.config.group_size,  # 4 sequences
        do_sample=True,
        pad_token_id=self.tokenizer.pad_token_id,
    )
```

**Characteristics:**
- All 4 options generate simultaneously with shared KV cache
- Efficient: ~same time as generating 1 option
- Blocking: UI waits until all 4 complete before displaying anything
- Peak VRAM: ~4.3GB for KV cache alone (8B model, 2048 context)

---

## VRAM Analysis

KV cache formula: `2 × num_layers × hidden_dim × seq_length × batch_size × 2 bytes`

| Model Size | Batch=4 (current) | Batch=1 (sequential) | Savings |
|------------|-------------------|----------------------|---------|
| 8B (32L, 4096H) | ~4.3 GB | ~1.1 GB | ~3.2 GB |
| 3B (26L, 2048H) | ~1.4 GB | ~0.35 GB | ~1.0 GB |
| 1B (16L, 2048H) | ~0.5 GB | ~0.13 GB | ~0.4 GB |

*Assumes 2048 max sequence length, fp16 KV cache*

---

## Streaming Options

### Option A: Sequential Streaming (Recommended)

Generate options one at a time, streaming each:

```python
def generate_options_streaming(self, prompt: str):
    """Generator yielding partial options as they generate."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    formatted_prompt = self._format_prompt(prompt)
    messages = self._build_messages(formatted_prompt)
    inputs = self.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(self.device)
    prompt_len = inputs.shape[1]

    all_options = []
    all_sequences = []

    for i in range(self.config.group_size):
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "min_p": self.config.min_p,
            "num_return_sequences": 1,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        # Capture sequences from thread
        result = {"sequences": None}
        def generate_with_capture():
            result["sequences"] = self.model.generate(**gen_kwargs)

        thread = Thread(target=generate_with_capture)
        thread.start()

        partial = ""
        for new_text in streamer:
            partial += new_text
            # Yield current state: which option, partial text, all options so far
            current_options = all_options + [partial] + [""] * (self.config.group_size - i - 1)
            yield (i, current_options)

        thread.join()
        all_options.append(partial)
        all_sequences.append(result["sequences"])

    # Store for training
    self.current_batch = {
        "prompt": prompt,
        "prompt_len": prompt_len,
        "sequences": torch.cat(all_sequences, dim=0),
    }

    yield (self.config.group_size, all_options)  # Final signal
```

**Pros:**
- Simple implementation using standard `TextIteratorStreamer`
- ~3GB less peak VRAM
- Tokens appear immediately (better perceived responsiveness)
- Each option streams cleanly into its own textbox

**Cons:**
- ~4x total wall-clock time (each option generated separately)
- Loses batch efficiency (no shared attention computation)

---

### Option B: Parallel Batched Streaming (Complex)

Keep `num_return_sequences=4` but add streaming. The challenge is that `TextIteratorStreamer` with batched generation interleaves tokens from all sequences.

**Approach:** Custom streamer that tracks token IDs per sequence:

```python
class BatchedTokenStreamer:
    """Streamer that separates tokens by sequence for batched generation."""

    def __init__(self, tokenizer, num_sequences):
        self.tokenizer = tokenizer
        self.num_sequences = num_sequences
        self.token_buffers = [[] for _ in range(num_sequences)]
        self.queue = Queue()
        self.stop_signal = object()

    def put(self, token_ids):
        """Called by generate() with shape [batch_size, 1] for each step."""
        # token_ids has shape [num_sequences, 1]
        for i in range(self.num_sequences):
            self.token_buffers[i].append(token_ids[i, 0].item())

        # Decode each sequence's current state
        decoded = []
        for buffer in self.token_buffers:
            text = self.tokenizer.decode(buffer, skip_special_tokens=True)
            decoded.append(text)

        self.queue.put(decoded)

    def end(self):
        self.queue.put(self.stop_signal)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is self.stop_signal:
                break
            yield item
```

**Pros:**
- Maintains batch efficiency
- All 4 options stream simultaneously

**Cons:**
- Complex implementation
- Higher peak VRAM
- May have edge cases with different tokenizers
- `model.generate()` may not call streamer's `put()` with the expected shape for all models

---

### Option C: Hybrid Approach

Generate 2 batches of 2 sequences each, streaming each batch:

- First batch: Options A & B stream together
- Second batch: Options C & D stream together

**Pros:**
- Middle ground on efficiency vs complexity
- ~2GB VRAM savings vs full batch

**Cons:**
- Still need to handle batched streamer output
- More complex than sequential, less efficient than full parallel

---

## UI Integration

The UI already supports generators for progressive updates. The `stream_next_prompt_and_options()` function would change to:

```python
def stream_next_prompt_and_options():
    """Stream prompt immediately, then stream options."""
    # ... get prompt ...

    stats = app.get_stats()

    # Yield prompt with empty options
    yield (stats[0], stats[1], stats[2], prompt_text, "", "", "", "", "")

    # Stream options
    for option_idx, partial_options in app.engine.generate_options_streaming(prompt):
        yield (
            stats[0], stats[1], stats[2], prompt_text,
            partial_options[0],
            partial_options[1],
            partial_options[2],
            partial_options[3],
            "",
        )
```

---

## Recommendation

**Use Sequential Streaming (Option A)** because:

1. **Simpler implementation** - Uses standard HuggingFace APIs without custom streamer logic
2. **Lower VRAM** - ~3GB headroom helps on consumer GPUs (8GB, 12GB cards)
3. **Better UX** - Seeing tokens stream immediately feels more responsive than waiting 8-10 seconds for all 4 to appear at once
4. **Easier debugging** - Each option generates independently, easier to trace issues
5. **Training compatibility** - Can still capture sequences for `current_batch` without threading complexity

The ~4x slowdown in total generation time is acceptable because:
- Streaming provides immediate feedback
- User can start reading Option A while B/C/D generate
- Interactive preference tuning is not latency-critical like chat

---

## Files to Modify

1. **`engine/unsloth_engine.py`** - Add `generate_options_streaming()` method
2. **`engine/base.py`** - Add abstract method signature (optional, for interface consistency)
3. **`ui/app.py`** - Update `stream_next_prompt_and_options()` to use streaming generator
