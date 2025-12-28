"""
TRL/HuggingFace Engine
======================

Training engine using standard HuggingFace Transformers + PEFT + TRL (optional).
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from threading import Thread
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from .base import BaseEngine
from .utils import BatchedTextStreamer

if TYPE_CHECKING:
    from config import TrellisConfig


class TRLEngine(BaseEngine):
    """
    Training engine using standard HF transformers + PEFT.
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
        return "TRL (LoRA)"

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self) -> str:
        """Initialize model, tokenizer, LoRA, optimizer."""
        print(f"Loading model {self.config.model_name} with TRL engine...")

        # Quantization Config
        bnb_config = None
        torch_dtype = torch.float16
        
        if self.config.load_in_4bit:
            print("Using 4-bit quantization (BitsAndBytes)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
        elif self.config.load_in_8bit:
            print("Using 8-bit quantization (BitsAndBytes)")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                 torch_dtype = torch.bfloat16

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        self._ensure_pad_token()

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.config.load_in_4bit or self.config.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Setup LoRA
        peft_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(self.config.lora_target_modules),
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        # Persistent optimizer
        # Only optimize parameters that require gradients (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self._set_inference_mode()
        print("Model ready.")
        return "Model ready."

    def _ensure_pad_token(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Also ensure padding side is right (usually) but for generation left is better.
        # However, for training/batched inference standardizing is key.
        self.tokenizer.padding_side = "left"

    def _ensure_chat_template(self):
        """Provide a default chat template if missing."""
        if getattr(self.tokenizer, "chat_template", None):
            return
        # Simple fallback template
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "User: {{ message['content'] }}\\n"
            "{% elif message['role'] == 'assistant' %}"
            "Assistant: {{ message['content'] }}\\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
        )

    def _format_prompt(self, prompt: str) -> str:
        """Apply system prompt and prefix/suffix if configured."""
        formatted = prompt
        if self.config.prompt_prefix:
            formatted = self.config.prompt_prefix + formatted
        if self.config.prompt_suffix:
            formatted = formatted + self.config.prompt_suffix
        return formatted

    def _set_inference_mode(self) -> None:
        if self.model is None:
            return
        self.model.eval()

    def _set_training_mode(self) -> None:
        if self.model is None:
            return
        self.model.train()

    def _build_inputs(self, prompt: str) -> tuple[torch.Tensor, int, str]:
        """Construct input ids with optional think tag and return prompt length."""
        formatted_prompt = self._format_prompt(prompt)

        if self.config.use_chat_template:
            self._ensure_chat_template()
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
        else:
            # Base model mode
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to(self.device)

        if self.config.append_think_tag and self.config.think_tag:
            think_tokens = self.tokenizer.encode(
                self.config.think_tag,
                add_special_tokens=False,
            )
            if think_tokens:
                think_tensor = torch.tensor([think_tokens], device=self.device)
                inputs = torch.cat([inputs, think_tensor], dim=1)

        prompt_len = inputs.shape[1]
        return inputs, prompt_len, formatted_prompt

    def generate_options(self, prompt: str) -> list[str]:
        """Generate GROUP_SIZE continuations for the given prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._set_inference_mode()

        inputs, prompt_len, _ = self._build_inputs(prompt)

        with torch.no_grad():
            sequences = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                # min_p is supported in newer transformers, checking fallback
                top_p=0.9 if not hasattr(self.config, 'min_p') else 1.0, 
                # passing min_p via generation_config or kwargs if transformers supports it
                # For safety, standard sampling:
                do_sample=True,
                num_return_sequences=self.config.group_size,
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

    def generate_options_streaming(self, prompt: str):
        """Stream GROUP_SIZE continuations in parallel."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._set_inference_mode()

        inputs, prompt_len, _ = self._build_inputs(prompt)

        stop_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            stop_ids.add(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            stop_ids.add(self.tokenizer.pad_token_id)

        streamer = BatchedTextStreamer(
            self.tokenizer,
            batch_size=self.config.group_size,
            skip_prompt_tokens=prompt_len,
            stop_token_ids=stop_ids,
        )

        gen_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "do_sample": True,
            "num_return_sequences": self.config.group_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }

        # Handle min_p if available in config and transformers
        if hasattr(self.config, 'min_p') and self.config.min_p > 0:
            # Note: min_p support depends on transformers version, generally safe to pass as kwarg
            gen_kwargs["min_p"] = self.config.min_p

        result = {"sequences": None}

        def generate_with_capture():
            with torch.no_grad():
                result["sequences"] = self.model.generate(**gen_kwargs)

        thread = Thread(target=generate_with_capture)
        thread.start()

        final_options: list[str] = []

        for decoded in streamer:
            padded = (decoded + [""] * self.config.group_size)[:self.config.group_size]
            final_options = padded
            yield padded

        thread.join()
        sequences = result["sequences"]
        if sequences is None:
            return

        self.current_batch = {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "sequences": sequences,
        }

        continuations = sequences[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(continuations, skip_special_tokens=True)
        decoded = (decoded + [""] * self.config.group_size)[:self.config.group_size]

        # Emit final decoded options to ensure UI has full texts
        if decoded != final_options:
            yield decoded

    def train_step(self, choice_idx: int) -> tuple[str, dict]:
        """
        Perform one preference update.
        """
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model not loaded")
        if self.current_batch is None:
            raise RuntimeError("No generations to train on")

        prompt_len = self.current_batch["prompt_len"]
        input_ids = self.current_batch["sequences"].to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Trim
        lengths = attention_mask.sum(dim=1)
        max_len = int(lengths.max().item())
        max_len = max(max_len, prompt_len + 1)
        max_len = min(max_len, input_ids.shape[1])
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        prompt_len = min(prompt_len, max_len)

        advantages = self._compute_advantages(choice_idx)

        self._set_training_mode()
        self.optimizer.zero_grad()

        # Policy logprobs
        seq_logp, token_logp, token_mask = self._masked_seq_logp(
            self.model, input_ids, attention_mask, prompt_len
        )

        # Policy gradient loss
        pg_loss = -(advantages * seq_logp).mean()

        # Optional KL anchor
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.config.kl_beta > 0.0:
            # With PEFT, we can disable adapters contextually
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

        self._set_inference_mode()

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

        # Run forward pass
        # Note: PEFT models forward usually works like standard models
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
        """L2 distance of LoRA weights from zero."""
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
        # Helper to move to cpu
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
        """Save current adapter to disk."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)

    def merge_and_save(self, path: str) -> str:
        """Merge LoRA into base model and save."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            print("Merging LoRA into base model...")
            merged_model = self.model.merge_and_unload()

            print(f"Saving merged model to {path}...")
            os.makedirs(path, exist_ok=True)
            merged_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

            return f"Merged model saved to {path}"
        except Exception as e:
            return f"Error merging model: {e}"
