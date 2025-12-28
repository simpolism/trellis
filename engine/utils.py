from __future__ import annotations

import torch
from queue import Queue
from typing import Optional

class BatchedTextStreamer:
    """Simple streamer to keep batched token streams separated."""

    def __init__(
        self,
        tokenizer,
        batch_size: int,
        skip_prompt_tokens: int = 0,
        stop_token_ids: Optional[set[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.skip_prompt_tokens = max(0, skip_prompt_tokens)
        self.stop_token_ids = stop_token_ids or set()

        self.token_buffers: list[list[int]] = [[] for _ in range(batch_size)]
        self.tokens_seen: list[int] = [0 for _ in range(batch_size)]
        self.finished: list[bool] = [False for _ in range(batch_size)]

        self.queue: Queue[list[str]] = Queue()
        self.stop_signal = object()

    def put(self, value):
        # value can be torch.Tensor, list[list[int]], or list[int]
        if torch.is_tensor(value):
            tokens = value.detach().cpu().tolist()
        else:
            tokens = value

        if not tokens:
            return

        if isinstance(tokens[0], int):
            tokens = [[t] for t in tokens]

        appended = False

        for idx in range(min(self.batch_size, len(tokens))):
            for token_id in tokens[idx]:
                if self.finished[idx]:
                    continue

                # Skip prompt tokens if requested
                if self.tokens_seen[idx] < self.skip_prompt_tokens:
                    self.tokens_seen[idx] += 1
                    continue

                if self.stop_token_ids and token_id in self.stop_token_ids:
                    self.finished[idx] = True
                    continue

                self.token_buffers[idx].append(int(token_id))
                appended = True

        # Avoid flooding queue with empty updates (e.g., skipping prompt tokens)
        if not appended:
            return

        decoded = [
            self.tokenizer.decode(buf, skip_special_tokens=True) if buf else ""
            for buf in self.token_buffers
        ]
        self.queue.put(decoded)

    def end(self):
        self.queue.put(self.stop_signal)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is self.stop_signal:
                break
            yield item
