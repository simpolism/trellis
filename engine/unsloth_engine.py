"""
UnSloth Engine
==============

Training engine using UnSloth for 4-bit quantized models with LoRA.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from .base import BaseEngine, VRAMEstimate
from .vram import estimate_vram_unsloth

if TYPE_CHECKING:
    from config import TrellisConfig


class UnslothEngine(BaseEngine):
    """
    Training engine using UnSloth for efficient 4-bit inference and LoRA training.
    """

    def __init__(self, config: "TrellisConfig"):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None
        self.optimizer = None

        # Current generation batch (for training)
        self.current_batch: Optional[dict] = None

    @property
    def name(self) -> str:
        return "UnSloth (4-bit + LoRA)"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> str:
        """Initialize model, tokenizer, LoRA, optimizer."""
        from unsloth import FastLanguageModel

        print("Loading model...")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_rank,
            target_modules=list(self.config.lora_target_modules),
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407,
        )

        self._ensure_pad_token()

        # Cast LoRA params to fp32 for training stability
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        # Persistent optimizer
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.model.eval()
        print("Model ready.")
        return "Model ready."

    def _ensure_pad_token(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_prompt(self, prompt: str) -> str:
        """Apply system prompt and prefix/suffix if configured."""
        # Apply prefix and suffix
        formatted = prompt
        if self.config.prompt_prefix:
            formatted = self.config.prompt_prefix + formatted
        if self.config.prompt_suffix:
            formatted = formatted + self.config.prompt_suffix
        return formatted

    def generate_options(self, prompt: str) -> list[str]:
        """Generate GROUP_SIZE continuations for the given prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.model.eval()

        # Format prompt with prefix/suffix
        formatted_prompt = self._format_prompt(prompt)

        # Build messages
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": formatted_prompt})

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        prompt_len = inputs.shape[1]

        with torch.no_grad():
            if self.config.sequential_streaming:
                # Sequential generation: generate one sequence at a time
                # This reduces VRAM usage by only maintaining one KV cache at a time
                all_sequences = []
                for _ in range(self.config.group_size):
                    sequence = self.model.generate(
                        inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        min_p=self.config.min_p,
                        num_return_sequences=1,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    all_sequences.append(sequence)
                    # Free GPU memory after each generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Concatenate all sequences into a single batch tensor
                sequences = torch.cat(all_sequences, dim=0)
            else:
                # Parallel generation: generate all sequences at once
                sequences = self.model.generate(
                    inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    min_p=self.config.min_p,
                    num_return_sequences=self.config.group_size,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # Store for training
        self.current_batch = {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "sequences": sequences,
        }

        # Decode continuations
        continuations = sequences[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(continuations, skip_special_tokens=True)

        return (decoded + [""] * self.config.group_size)[:self.config.group_size]

    def train_step(self, choice_idx: int) -> tuple[str, dict]:
        """
        Perform one preference update.

        Args:
            choice_idx: 0..GROUP_SIZE-1 for chosen, GROUP_SIZE for reject-all

        Returns:
            (status_message, metrics_dict)
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model not loaded")
        if self.current_batch is None:
            raise RuntimeError("No generations to train on")

        prompt_len = self.current_batch["prompt_len"]
        input_ids = self.current_batch["sequences"].to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        advantages = self._compute_advantages(choice_idx)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Policy logprobs
        seq_logp, token_logp, token_mask = self._masked_seq_logp(
            self.model, input_ids, attention_mask, prompt_len
        )

        # Policy gradient loss
        pg_loss = -(advantages * seq_logp).mean()

        # Optional KL anchor
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.config.kl_beta > 0.0 and hasattr(self.model, "disable_adapter"):
            with torch.no_grad():
                with self.model.disable_adapter():
                    _, token_logp_ref, _ = self._masked_seq_logp(
                        self.model, input_ids, attention_mask, prompt_len
                    )
            denom = token_mask.sum().clamp(min=1)
            kl_est = (token_logp - token_logp_ref).sum() / denom
            kl_loss = kl_est ** 2

        loss = pg_loss + self.config.kl_beta * kl_loss
        loss.backward()
        self.optimizer.step()

        self.model.eval()

        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "total_loss": loss.item(),
        }

        msg = f"PG: {metrics['pg_loss']:.4f}"
        if self.config.kl_beta > 0:
            msg += f" | KL: {self.config.kl_beta * metrics['kl_loss']:.4f}"
        msg += f" | Total: {metrics['total_loss']:.4f}"

        return msg, metrics

    def _compute_advantages(self, choice_idx: int) -> torch.Tensor:
        """Convert choice index to advantage vector."""
        n = self.config.group_size

        if choice_idx == n:  # Reject all
            return torch.full((n,), -1.0, device=self.device)

        rewards = torch.zeros(n, device=self.device)
        rewards[choice_idx] = 1.0

        mean = rewards.mean()
        std = rewards.std(unbiased=False) + 1e-8
        return (rewards - mean) / std

    def _masked_seq_logp(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
    ):
        """Compute per-sequence and per-token log-probs for continuations."""
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        token_mask = (shift_labels != -100)

        logp_all = F.log_softmax(shift_logits, dim=-1)

        safe_labels = shift_labels.clone()
        safe_labels[~token_mask] = 0

        token_logp = logp_all.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp_masked = token_logp * token_mask

        seq_logp = token_logp_masked.sum(dim=1)
        return seq_logp, token_logp_masked, token_mask

    def compute_drift(self) -> float:
        """L2 distance of LoRA weights from zero (i.e., from base model)."""
        if self.model is None:
            return 0.0

        total = 0.0
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                total += param.detach().float().pow(2).sum().item()
        return math.sqrt(total)

    def get_adapter_state(self) -> dict:
        """Snapshot adapter weights (CPU)."""
        from peft.utils import get_peft_model_state_dict
        sd = get_peft_model_state_dict(self.model)
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def get_optimizer_state(self) -> dict:
        """Snapshot optimizer state (CPU)."""
        state = self.optimizer.state_dict()

        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.cpu().clone()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu(x) for x in obj]
            return obj

        return to_cpu(state)

    def restore_state(self, adapter_state: dict, optimizer_state: dict) -> None:
        """Restore adapter and optimizer from snapshots."""
        from peft.utils import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, adapter_state)

        if optimizer_state:
            def to_device(obj):
                if torch.is_tensor(obj):
                    return obj.to(self.device)
                elif isinstance(obj, dict):
                    return {k: to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_device(x) for x in obj]
                return obj

            self.optimizer.load_state_dict(to_device(optimizer_state))

    def save_adapter(self, path: str) -> None:
        """Save current adapter to disk in HuggingFace format."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    def merge_and_save(self, path: str) -> str:
        """
        Merge LoRA into base model and save full model.

        Note: This temporarily increases memory usage significantly.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # UnSloth provides a merge method
            from unsloth import FastLanguageModel

            print("Merging LoRA into base model...")

            # Get merged model
            merged_model = self.model.merge_and_unload()

            print(f"Saving merged model to {path}...")
            os.makedirs(path, exist_ok=True)
            merged_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            return f"Merged model saved to {path}"

        except Exception as e:
            return f"Error merging model: {e}"

    @classmethod
    def estimate_vram(cls, config: "TrellisConfig") -> VRAMEstimate:
        """Estimate VRAM requirements before loading."""
        return estimate_vram_unsloth(config)
