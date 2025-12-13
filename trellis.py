"""
TRELLIS — Interactive Preference Steering with Branching State
==============================================================

A framework for guided growth of model personality through direct preference.
Generate responses, pick the one with the right vibe, update weights, branch
to explore alternatives. Think of it as version control for model disposition.

Core loop:
    prompt → sample variants → vibe-check → update LoRA → observe shift

The state tree lets you:
    - Branch before risky updates
    - Explore divergent personality directions  
    - Compare branches side-by-side
    - Checkout any previous state (with optimizer momentum intact)

Hardware target: RTX 4070 Ti Super (16 GB) with 4-bit base + LoRA.

Usage:
    python trellis.py                    # Launch Gradio UI
    python trellis.py --dry-run          # UI without GPU (for testing)
"""

from __future__ import annotations

import math
import os
import random
import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse

import torch
import torch.nn.functional as F
import gradio as gr


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrellisConfig:
    """All tunable parameters in one place."""
    
    # Model
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length: int = 2048
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    
    # Generation
    group_size: int = 4
    max_new_tokens: int = 256
    temperature: float = 1.2
    min_p: float = 0.1
    
    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    kl_beta: float = 0.03  # KL anchor strength; 0 disables
    
    # Paths
    save_dir: str = "./trellis_states"
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrellisConfig":
        with open(path) as f:
            return cls(**json.load(f))


# =============================================================================
# PROMPT SOURCE
# =============================================================================

class PromptSource:
    """
    Loads prompts from HuggingFace datasets.
    
    Attempts to auto-detect the text column, or uses explicit column name.
    Supports shuffling and subset selection.
    """
    
    # Common column names for prompts in various datasets
    CANDIDATE_COLUMNS = [
        "question", "prompt", "instruction", "input", "text", 
        "query", "content", "message", "human", "user",
    ]
    
    def __init__(self):
        self.dataset = None
        self.dataset_id: Optional[str] = None
        self.prompts: list[str] = []
        self.index: int = 0
        self.text_column: Optional[str] = None
    
    def load(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: str = "train",
        text_column: Optional[str] = None,
        shuffle: bool = True,
        max_prompts: Optional[int] = None,
    ) -> str:
        """
        Load a dataset from HuggingFace.
        
        Args:
            dataset_id: HF dataset ID (e.g., "Anthropic/model-written-evals")
            subset: Dataset subset/config name (e.g., "persona")
            split: Which split to use (default: "train")
            text_column: Column containing prompts (auto-detected if None)
            shuffle: Whether to shuffle prompts
            max_prompts: Limit number of prompts (None for all)
        
        Returns:
            Status message
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return "⚠️ Install datasets: pip install datasets"
        
        try:
            # Load dataset
            if subset:
                ds = load_dataset(dataset_id, subset, split=split)
            else:
                ds = load_dataset(dataset_id, split=split)
            
            self.dataset = ds
            self.dataset_id = dataset_id
            
            # Auto-detect or validate text column
            columns = ds.column_names
            
            if text_column:
                if text_column not in columns:
                    return f"⚠️ Column '{text_column}' not found. Available: {columns}"
                self.text_column = text_column
            else:
                # Try to auto-detect
                self.text_column = self._detect_text_column(columns)
                if not self.text_column:
                    return f"⚠️ Could not detect text column. Available: {columns}. Specify manually."
            
            # Extract prompts
            self.prompts = [str(row[self.text_column]) for row in ds if row[self.text_column]]
            
            # Limit if requested
            if max_prompts and len(self.prompts) > max_prompts:
                self.prompts = self.prompts[:max_prompts]
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(self.prompts)
            
            self.index = 0
            
            subset_str = f"/{subset}" if subset else ""
            return f"✅ Loaded {len(self.prompts)} prompts from {dataset_id}{subset_str} (column: {self.text_column})"
            
        except Exception as e:
            return f"⚠️ Failed to load dataset: {e}"
    
    def _detect_text_column(self, columns: list[str]) -> Optional[str]:
        """Try to find the text column automatically."""
        # Check exact matches first (case-insensitive)
        columns_lower = {c.lower(): c for c in columns}
        for candidate in self.CANDIDATE_COLUMNS:
            if candidate in columns_lower:
                return columns_lower[candidate]
        
        # Check partial matches
        for candidate in self.CANDIDATE_COLUMNS:
            for col in columns:
                if candidate in col.lower():
                    return col
        
        # Fall back to first string-looking column
        return columns[0] if columns else None
    
    def next(self) -> Optional[str]:
        """Get the next prompt, cycling if exhausted."""
        if not self.prompts:
            return None
        
        prompt = self.prompts[self.index]
        self.index = (self.index + 1) % len(self.prompts)
        return prompt
    
    def peek(self) -> Optional[str]:
        """See current prompt without advancing."""
        if not self.prompts:
            return None
        return self.prompts[self.index]
    
    def skip(self, n: int = 1):
        """Skip forward n prompts."""
        if self.prompts:
            self.index = (self.index + n) % len(self.prompts)
    
    def reset(self, shuffle: bool = True):
        """Reset to beginning, optionally reshuffling."""
        if shuffle and self.prompts:
            random.shuffle(self.prompts)
        self.index = 0
    
    def remaining(self) -> int:
        """How many prompts until we cycle."""
        return len(self.prompts) - self.index if self.prompts else 0
    
    def total(self) -> int:
        """Total number of prompts."""
        return len(self.prompts)
    
    def is_loaded(self) -> bool:
        """Whether a dataset is loaded."""
        return len(self.prompts) > 0
    
    def status(self) -> str:
        """Current status string."""
        if not self.prompts:
            return "No dataset loaded"
        return f"{self.dataset_id}: {self.index + 1}/{len(self.prompts)}"


# =============================================================================
# STATE TREE
# =============================================================================

@dataclass
class TreeNode:
    """A snapshot of model state at a point in the exploration."""
    
    id: str
    name: str
    parent_id: Optional[str]
    created_at: str
    step_count: int
    
    # What led here (for history display)
    prompt: Optional[str] = None
    choice: Optional[str] = None
    chosen_response: Optional[str] = None  # The actual model output that was selected
    
    # Metrics at this point
    drift_from_root: float = 0.0
    
    # State tensors (CPU, populated on save)
    adapter_state: Optional[dict] = field(default=None, repr=False)
    optimizer_state: Optional[dict] = field(default=None, repr=False)


class Journal:
    """
    Dual-mode logging for Trellis sessions.
    
    - Session log: Chronological record of everything (branches, checkouts, trains)
    - Lineage narrative: Clean story from root to a specific node (for writeups)
    """
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = save_dir / "session_log.md"
        self._init_session_log()
    
    def _init_session_log(self):
        """Start a new session log with timestamp."""
        header = f"""# Trellis Session Log
*Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

"""
        with open(self.session_path, "w") as f:
            f.write(header)
    
    def _append(self, text: str):
        """Append to session log."""
        with open(self.session_path, "a") as f:
            f.write(text + "\n")
    
    def log_init(self, model_name: str):
        """Log model initialization."""
        self._append(f"🚀 **Loaded model:** `{model_name}`\n")
    
    def log_generation(self, prompt: str, options: list[str]):
        """Log a generation event."""
        self._append(f"⚡ **Generated options for:**\n> {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n")
    
    def log_train(
        self, 
        step: int,
        node_name: str,
        prompt: str,
        choice: str,
        chosen_text: str,
        drift: float,
        metrics: dict,
    ):
        """Log a training step."""
        entry = f"""### Step {step} → `{node_name}`
**Prompt:**
> {prompt[:300]}{'...' if len(prompt) > 300 else ''}

**Choice:** {choice} | **Drift:** {drift:.3f} | **Loss:** {metrics.get('total_loss', 0):.4f}

<details>
<summary>Selected response</summary>

{chosen_text[:500]}{'...' if len(chosen_text) > 500 else ''}

</details>

"""
        self._append(entry)
    
    def log_branch(self, branch_name: str, from_node: str):
        """Log branch creation."""
        self._append(f"""---
🌿 **Branched:** `{branch_name}` from `{from_node}`

---
""")
    
    def log_checkout(self, node_name: str):
        """Log a checkout."""
        self._append(f"""---
↩️ **Checked out:** `{node_name}`

---
""")
    
    def log_reject_all(self, step: int, node_name: str, prompt: str, drift: float):
        """Log a reject-all training step."""
        entry = f"""### Step {step} → `{node_name}` *(reject all)*
**Prompt:**
> {prompt[:300]}{'...' if len(prompt) > 300 else ''}

**Choice:** Reject All (pushed down all options) | **Drift:** {drift:.3f}

"""
        self._append(entry)
    
    def generate_lineage_narrative(self, tree: "StateTree", node_id: Optional[str] = None) -> str:
        """
        Generate a clean narrative from root to the specified node.
        This is the "story" of a branch, suitable for blog posts.
        """
        lineage = tree.get_lineage(node_id)
        
        if not lineage:
            return "*No history yet.*"
        
        current = lineage[-1] if lineage else None
        title = current.name if current else "unknown"
        
        lines = [
            f"# Lineage: {title}",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            f"A narrative of the choices that shaped this branch.",
            "",
            "---",
            "",
        ]
        
        # Skip root (step 0, no choices made)
        training_steps = [n for n in lineage if n.step_count > 0 and n.choice]
        
        if not training_steps:
            lines.append("*No training steps yet — this is the base model.*")
        else:
            for i, node in enumerate(training_steps, 1):
                prompt_preview = node.prompt[:300] + "..." if node.prompt and len(node.prompt) > 300 else (node.prompt or "")
                
                lines.append(f"## Step {i}: {node.name}")
                lines.append("")
                
                # Prompt
                if node.prompt:
                    lines.append("**Prompt:**")
                    lines.append(f"> {prompt_preview}")
                    lines.append("")
                
                # Choice label
                lines.append(f"**Selected:** {node.choice}")
                lines.append("")
                
                # The actual response that was chosen
                if node.chosen_response:
                    response_preview = node.chosen_response[:600]
                    if len(node.chosen_response) > 600:
                        response_preview += "..."
                    lines.append("**Response:**")
                    lines.append("")
                    # Indent response as a quote block
                    for response_line in response_preview.split('\n'):
                        lines.append(f"> {response_line}")
                    lines.append("")
                elif node.choice == "Reject All":
                    lines.append("*(All options rejected — pushed probability mass away from sampled responses)*")
                    lines.append("")
                
                lines.append(f"*Drift from base: {node.drift_from_root:.3f}*")
                lines.append("")
                lines.append("---")
                lines.append("")
        
        # Summary
        if current:
            lines.extend([
                "## Summary",
                "",
                f"- **Total steps:** {current.step_count}",
                f"- **Final drift:** {current.drift_from_root:.3f}",
                f"- **Branch:** `{current.name}`",
            ])
        
        return "\n".join(lines)
    
    def save_lineage(self, tree: "StateTree", node_id: Optional[str] = None, filename: Optional[str] = None):
        """Save lineage narrative to a file."""
        narrative = self.generate_lineage_narrative(tree, node_id)
        
        node = tree.nodes.get(node_id or tree.current_id)
        if filename is None:
            safe_name = (node.name if node else "lineage").replace(" ", "_").replace("/", "-")
            filename = f"lineage_{safe_name}.md"
        
        path = self.save_dir / filename
        with open(path, "w") as f:
            f.write(narrative)
        
        return path


class StateTree:
    """
    Version control for model personality.
    
    Nodes form a tree rooted at the initial (untrained) state.
    You can branch, checkout, compare, and persist to disk.
    """
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.nodes: dict[str, TreeNode] = {}
        self.current_id: Optional[str] = None
    
    @property
    def current(self) -> Optional[TreeNode]:
        return self.nodes.get(self.current_id) if self.current_id else None
    
    def create_root(self, adapter_state: dict, optimizer_state: dict) -> TreeNode:
        """Initialize the tree with the base (untrained) state."""
        node = TreeNode(
            id=self._make_id(),
            name="root",
            parent_id=None,
            created_at=self._timestamp(),
            step_count=0,
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )
        self.nodes[node.id] = node
        self.current_id = node.id
        return node
    
    def commit(
        self,
        adapter_state: dict,
        optimizer_state: dict,
        step_count: int,
        drift: float,
        prompt: Optional[str] = None,
        choice: Optional[str] = None,
        chosen_response: Optional[str] = None,
        name: Optional[str] = None,
    ) -> TreeNode:
        """
        Save current state as a new node, child of current.
        Automatically advances current_id to the new node.
        """
        node = TreeNode(
            id=self._make_id(),
            name=name or f"step-{step_count}",
            parent_id=self.current_id,
            created_at=self._timestamp(),
            step_count=step_count,
            prompt=prompt,
            choice=choice,
            chosen_response=chosen_response,
            drift_from_root=drift,
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )
        self.nodes[node.id] = node
        self.current_id = node.id
        return node
    
    def branch(self, name: str) -> TreeNode:
        """
        Create a named branch point at current state (no new state, just a marker).
        Useful for marking "I might want to come back here."
        """
        if not self.current:
            raise ValueError("No current node to branch from")
        
        # Clone current node with new name
        cur = self.current
        node = TreeNode(
            id=self._make_id(),
            name=name,
            parent_id=cur.parent_id,
            created_at=self._timestamp(),
            step_count=cur.step_count,
            prompt=cur.prompt,
            choice=cur.choice,
            chosen_response=cur.chosen_response,
            drift_from_root=cur.drift_from_root,
            adapter_state=cur.adapter_state,  # shared reference is fine (immutable use)
            optimizer_state=cur.optimizer_state,
        )
        self.nodes[node.id] = node
        self.current_id = node.id
        return node
    
    def checkout(self, node_id: str) -> TreeNode:
        """Switch to a different node. Returns the node for state restoration."""
        if node_id not in self.nodes:
            raise KeyError(f"Unknown node: {node_id}")
        self.current_id = node_id
        return self.nodes[node_id]
    
    def rename(self, node_id: str, new_name: str):
        """Rename a node."""
        if node_id in self.nodes:
            self.nodes[node_id].name = new_name
    
    def get_lineage(self, node_id: Optional[str] = None) -> list[TreeNode]:
        """Get ancestors from root to the given node."""
        node_id = node_id or self.current_id
        lineage = []
        while node_id:
            node = self.nodes.get(node_id)
            if not node:
                break
            lineage.append(node)
            node_id = node.parent_id
        return list(reversed(lineage))
    
    def get_children(self, node_id: str) -> list[TreeNode]:
        """Get immediate children of a node."""
        return [n for n in self.nodes.values() if n.parent_id == node_id]
    
    def render_tree(self) -> str:
        """Render the tree as ASCII art for display."""
        if not self.nodes:
            return "(empty)"
        
        # Find root
        roots = [n for n in self.nodes.values() if n.parent_id is None]
        if not roots:
            return "(no root)"
        
        lines = []
        self._render_subtree(roots[0], "", True, lines)
        return "\n".join(lines)
    
    def _render_subtree(self, node: TreeNode, prefix: str, is_last: bool, lines: list):
        marker = "●" if node.id == self.current_id else "○"
        branch = "└─" if is_last else "├─"
        
        drift_str = f" (drift: {node.drift_from_root:.2f})" if node.drift_from_root > 0 else ""
        current_str = " ← current" if node.id == self.current_id else ""
        
        lines.append(f"{prefix}{branch}{marker} {node.name} [step {node.step_count}]{drift_str}{current_str}")
        
        children = self.get_children(node.id)
        for i, child in enumerate(children):
            ext = "  " if is_last else "│ "
            self._render_subtree(child, prefix + ext, i == len(children) - 1, lines)
    
    def list_nodes(self) -> list[tuple[str, str]]:
        """Return (id, display_name) pairs for UI dropdowns."""
        return [(n.id, f"{n.name} [step {n.step_count}]") for n in self.nodes.values()]
    
    def _make_id(self) -> str:
        return uuid.uuid4().hex[:8]
    
    def _timestamp(self) -> str:
        return datetime.now().isoformat()
    
    # Persistence
    def save_to_disk(self):
        """Persist tree metadata (not tensors) to JSON."""
        meta = {
            "current_id": self.current_id,
            "nodes": {}
        }
        for nid, node in self.nodes.items():
            meta["nodes"][nid] = {
                "id": node.id,
                "name": node.name,
                "parent_id": node.parent_id,
                "created_at": node.created_at,
                "step_count": node.step_count,
                "prompt": node.prompt,
                "choice": node.choice,
                "chosen_response": node.chosen_response,
                "drift_from_root": node.drift_from_root,
            }
        with open(self.save_dir / "tree.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    def save_node_state(self, node_id: str):
        """Save a specific node's tensors to disk."""
        node = self.nodes.get(node_id)
        if not node or not node.adapter_state:
            return
        
        node_dir = self.save_dir / node_id
        node_dir.mkdir(exist_ok=True)
        
        torch.save(node.adapter_state, node_dir / "adapter.pt")
        if node.optimizer_state:
            torch.save(node.optimizer_state, node_dir / "optimizer.pt")


# =============================================================================
# TRAINING ENGINE
# =============================================================================

class TrellisEngine:
    """
    Handles model loading, generation, and training updates.
    Separated from state management for clarity.
    """
    
    def __init__(self, config: TrellisConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.device = "cuda" if torch.cuda.is_available() and not dry_run else "cpu"
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        
        # Current generation batch (for training)
        self.current_batch: Optional[dict] = None
    
    def load_model(self) -> str:
        """Initialize model, tokenizer, LoRA, optimizer."""
        if self.dry_run:
            return "🧪 Dry-run mode (no GPU)"
        
        from unsloth import FastLanguageModel
        from peft.utils import get_peft_model_state_dict
        
        print("⏳ Loading model...")
        
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
        print("✅ Model ready.")
        return "✅ Model ready."
    
    def _ensure_pad_token(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_options(self, prompt: str) -> list[str]:
        """Generate GROUP_SIZE continuations for the given prompt."""
        if self.dry_run:
            return [f"[Dry-run response {i+1} to: {prompt[:50]}...]" 
                    for i in range(self.config.group_size)]
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        self.model.eval()
        
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        
        prompt_len = inputs.shape[1]
        
        with torch.no_grad():
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
        if self.dry_run:
            return "🧪 Dry-run: would train here", {"pg_loss": 0.0, "kl_loss": 0.0}
        
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
            msg += f" | KL×β: {self.config.kl_beta * metrics['kl_loss']:.4f}"
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
        if self.dry_run or self.model is None:
            return 0.0
        
        total = 0.0
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                total += param.detach().float().pow(2).sum().item()
        return math.sqrt(total)
    
    def get_adapter_state(self) -> dict:
        """Snapshot adapter weights (CPU)."""
        if self.dry_run:
            return {"dummy": torch.zeros(1)}
        
        from peft.utils import get_peft_model_state_dict
        sd = get_peft_model_state_dict(self.model)
        return {k: v.detach().cpu().clone() for k, v in sd.items()}
    
    def get_optimizer_state(self) -> dict:
        """Snapshot optimizer state (CPU)."""
        if self.dry_run:
            return {}
        
        state = self.optimizer.state_dict()
        # Deep copy tensors to CPU
        def to_cpu(obj):
            if torch.is_tensor(obj):
                return obj.cpu().clone()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu(x) for x in obj]
            return obj
        return to_cpu(state)
    
    def restore_state(self, adapter_state: dict, optimizer_state: dict):
        """Restore adapter and optimizer from snapshots."""
        if self.dry_run:
            return
        
        from peft.utils import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, adapter_state)
        
        if optimizer_state:
            # Move optimizer state back to device
            def to_device(obj):
                if torch.is_tensor(obj):
                    return obj.to(self.device)
                elif isinstance(obj, dict):
                    return {k: to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_device(x) for x in obj]
                return obj
            self.optimizer.load_state_dict(to_device(optimizer_state))
    
    def save_adapter(self, path: str):
        """Save current adapter to disk."""
        if self.dry_run:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)


# =============================================================================
# UI CONTROLLER
# =============================================================================

class TrellisApp:
    """Wires together the engine, state tree, and Gradio UI."""
    
    def __init__(self, config: TrellisConfig, dry_run: bool = False):
        self.config = config
        self.engine = TrellisEngine(config, dry_run=dry_run)
        self.tree = StateTree(config.save_dir)
        self.journal = Journal(Path(config.save_dir))
        self.prompt_source = PromptSource()
        self.step_count = 0
        self.last_prompt = None
        self.last_choice = None
        self.last_options: list[str] = []  # Store for journal logging
    
    def initialize(self) -> tuple[str, str]:
        """Load model and create root node."""
        status = self.engine.load_model()
        
        # Create root state
        self.tree.create_root(
            adapter_state=self.engine.get_adapter_state(),
            optimizer_state=self.engine.get_optimizer_state(),
        )
        self.step_count = 0
        
        # Log to journal
        self.journal.log_init(self.config.model_name)
        
        tree_display = self.tree.render_tree()
        return status, tree_display
    
    def generate(self, prompt: str) -> tuple[str, str, str, str, str]:
        """Generate options and return them for display."""
        if self.engine.model is None and not self.engine.dry_run:
            blank = "⚠️ Load the model first."
            return blank, blank, blank, blank, ""
        
        self.last_prompt = prompt
        options = self.engine.generate_options(prompt)
        self.last_options = options  # Store for journal logging
        
        # Log to journal
        self.journal.log_generation(prompt, options)
        
        return (
            f"**Option A:**\n\n{options[0]}",
            f"**Option B:**\n\n{options[1]}",
            f"**Option C:**\n\n{options[2]}",
            f"**Option D:**\n\n{options[3]}",
            "✅ Generated. Pick your favorite.",
        )
    
    def train(self, choice: str) -> tuple[str, str, str]:
        """Perform training step, commit to tree."""
        choice_map = {
            "Option A": 0,
            "Option B": 1,
            "Option C": 2,
            "Option D": 3,
            "Reject All": self.config.group_size,
        }
        
        if choice not in choice_map:
            return "⚠️ Select an option first.", self.tree.render_tree(), ""
        
        choice_idx = choice_map[choice]
        self.last_choice = choice
        
        try:
            msg, metrics = self.engine.train_step(choice_idx)
        except RuntimeError as e:
            return f"⚠️ {e}", self.tree.render_tree(), ""
        
        self.step_count += 1
        drift = self.engine.compute_drift()
        
        # Get the chosen response text (None for reject-all)
        chosen_text = None
        if choice_idx < len(self.last_options):
            chosen_text = self.last_options[choice_idx]
        
        # Commit new state to tree
        node = self.tree.commit(
            adapter_state=self.engine.get_adapter_state(),
            optimizer_state=self.engine.get_optimizer_state(),
            step_count=self.step_count,
            drift=drift,
            prompt=self.last_prompt,  # Store full prompt
            choice=choice,
            chosen_response=chosen_text,
        )
        
        # Log to journal
        if choice_idx == self.config.group_size:
            # Reject all
            self.journal.log_reject_all(
                step=self.step_count,
                node_name=node.name,
                prompt=self.last_prompt or "",
                drift=drift,
            )
        else:
            # Normal selection
            self.journal.log_train(
                step=self.step_count,
                node_name=node.name,
                prompt=self.last_prompt or "",
                choice=choice,
                chosen_text=chosen_text or "",
                drift=drift,
                metrics=metrics,
            )
        
        status = f"✅ Step {self.step_count} | {msg}"
        drift_display = f"**Drift from base:** {drift:.3f}"
        
        return status, self.tree.render_tree(), drift_display
    
    def retaste(self) -> tuple[str, str, str, str, str]:
        """Re-generate with same prompt to see effect of update."""
        if not self.last_prompt:
            blank = "⚠️ No previous prompt."
            return blank, blank, blank, blank, ""
        return self.generate(self.last_prompt)
    
    def branch(self, name: str) -> tuple[str, str]:
        """Create a named branch at current position."""
        if not name.strip():
            return "⚠️ Enter a branch name.", self.tree.render_tree()
        
        from_node = self.tree.current
        from_name = from_node.name if from_node else "unknown"
        
        self.tree.branch(name.strip())
        
        # Log to journal
        self.journal.log_branch(name.strip(), from_name)
        
        return f"✅ Created branch: {name}", self.tree.render_tree()
    
    def checkout(self, node_id: str) -> tuple[str, str, str]:
        """Switch to a different node."""
        try:
            node = self.tree.checkout(node_id)
        except KeyError:
            return "⚠️ Unknown node.", self.tree.render_tree(), ""
        
        if node.adapter_state:
            self.engine.restore_state(node.adapter_state, node.optimizer_state or {})
        self.step_count = node.step_count
        
        # Log to journal
        self.journal.log_checkout(node.name)
        
        drift_display = f"**Drift from base:** {node.drift_from_root:.3f}"
        return f"✅ Checked out: {node.name}", self.tree.render_tree(), drift_display
    
    def compare(self, node_id_a: str, node_id_b: str, prompt: str) -> tuple[str, str]:
        """Generate from two different nodes for comparison."""
        if not prompt.strip():
            return "⚠️ Enter a prompt for comparison.", ""
        
        original_id = self.tree.current_id
        results = []
        
        for nid in [node_id_a, node_id_b]:
            node = self.tree.checkout(nid)
            if node.adapter_state:
                self.engine.restore_state(node.adapter_state, node.optimizer_state or {})
            
            options = self.engine.generate_options(prompt)
            results.append((node.name, options[0]))  # Just take first generation
        
        # Restore original position
        self.tree.checkout(original_id)
        orig_node = self.tree.current
        if orig_node and orig_node.adapter_state:
            self.engine.restore_state(orig_node.adapter_state, orig_node.optimizer_state or {})
        
        output_a = f"**{results[0][0]}:**\n\n{results[0][1]}"
        output_b = f"**{results[1][0]}:**\n\n{results[1][1]}"
        
        return output_a, output_b
    
    def save_current(self, name: str) -> str:
        """Save current adapter to disk."""
        if not name.strip():
            return "⚠️ Enter a name."
        
        path = os.path.join(self.config.save_dir, name.strip())
        self.engine.save_adapter(path)
        self.tree.save_to_disk()
        
        if self.tree.current_id:
            self.tree.save_node_state(self.tree.current_id)
        
        return f"💾 Saved to {path}"
    
    def get_node_choices(self) -> list[tuple[str, str]]:
        """Get node list for dropdowns."""
        return self.tree.list_nodes()
    
    def get_lineage_narrative(self) -> str:
        """Get the lineage narrative for current branch."""
        return self.journal.generate_lineage_narrative(self.tree)
    
    def get_session_log(self) -> str:
        """Read the current session log."""
        try:
            with open(self.journal.session_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "*No session log yet.*"
    
    def export_lineage(self, filename: str) -> str:
        """Export lineage narrative to a file."""
        if not filename.strip():
            filename = None
        path = self.journal.save_lineage(self.tree, filename=filename)
        return f"📝 Exported lineage to `{path}`"
    
    # Dataset methods
    def load_dataset(
        self, 
        dataset_id: str, 
        subset: str, 
        split: str, 
        text_column: str,
        shuffle: bool,
        max_prompts: int,
    ) -> tuple[str, str]:
        """Load a HuggingFace dataset for prompts."""
        # Clean inputs
        subset = subset.strip() if subset.strip() else None
        text_column = text_column.strip() if text_column.strip() else None
        max_prompts = int(max_prompts) if max_prompts else None
        
        status = self.prompt_source.load(
            dataset_id=dataset_id.strip(),
            subset=subset,
            split=split.strip() or "train",
            text_column=text_column,
            shuffle=shuffle,
            max_prompts=max_prompts,
        )
        
        return status, self.prompt_source.status()
    
    def next_prompt(self) -> tuple[str, str]:
        """Get the next prompt from the dataset."""
        prompt = self.prompt_source.next()
        if prompt is None:
            return "", "No dataset loaded"
        return prompt, self.prompt_source.status()
    
    def skip_prompt(self) -> tuple[str, str]:
        """Skip to next prompt without using current one."""
        self.prompt_source.skip()
        prompt = self.prompt_source.peek()
        return prompt or "", self.prompt_source.status()


# =============================================================================
# GRADIO UI
# =============================================================================

def build_ui(app: TrellisApp) -> gr.Blocks:
    """Construct the Gradio interface."""
    
    config = app.config
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Trellis") as demo:
        gr.Markdown("# 🌿 Trellis")
        gr.Markdown(
            "Interactive preference steering with branching state.\n\n"
            f"**Model:** `{config.model_name}` | "
            f"**Group size:** {config.group_size} | "
            f"**KL β:** {config.kl_beta}"
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                status_box = gr.Textbox(
                    label="Status", 
                    value="Not loaded", 
                    interactive=False
                )
                drift_box = gr.Markdown("**Drift from base:** 0.000")
            
            with gr.Column(scale=1):
                load_btn = gr.Button("🚀 Load Model", variant="primary")
        
        with gr.Tabs():
            # === MAIN TAB ===
            with gr.Tab("Steer"):
                with gr.Row():
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        lines=3,
                        placeholder="Ask something to shape the model's disposition...",
                        scale=4,
                    )
                    with gr.Column(scale=1):
                        dataset_status = gr.Textbox(
                            label="Dataset", 
                            value="No dataset loaded",
                            interactive=False,
                        )
                        next_prompt_btn = gr.Button("📋 Next Prompt")
                        skip_prompt_btn = gr.Button("⏭️ Skip", size="sm")
                
                with gr.Row():
                    generate_btn = gr.Button("⚡ Generate Options")
                    retaste_btn = gr.Button("🔄 Re-taste (same prompt)")
                
                with gr.Row():
                    opt_a = gr.Markdown("**Option A:** *(generate first)*")
                    opt_b = gr.Markdown("**Option B:** *(generate first)*")
                
                with gr.Row():
                    opt_c = gr.Markdown("**Option C:** *(generate first)*")
                    opt_d = gr.Markdown("**Option D:** *(generate first)*")
                
                choice_radio = gr.Radio(
                    choices=["Option A", "Option B", "Option C", "Option D", "Reject All"],
                    label="Which has the right vibe?",
                )
                
                train_btn = gr.Button("💪 Train (Update Weights)", variant="primary")
            
            # === DATASET TAB ===
            with gr.Tab("Dataset"):
                gr.Markdown(
                    "Load prompts from a HuggingFace dataset. "
                    "The text column is auto-detected, or you can specify it manually."
                )
                
                with gr.Row():
                    ds_id_input = gr.Textbox(
                        label="Dataset ID",
                        placeholder="Anthropic/model-written-evals",
                        scale=2,
                    )
                    ds_subset_input = gr.Textbox(
                        label="Subset (optional)",
                        placeholder="persona",
                        scale=1,
                    )
                
                with gr.Row():
                    ds_split_input = gr.Textbox(
                        label="Split",
                        value="train",
                        scale=1,
                    )
                    ds_column_input = gr.Textbox(
                        label="Text Column (auto-detect if empty)",
                        placeholder="question",
                        scale=1,
                    )
                    ds_max_input = gr.Number(
                        label="Max Prompts (0=all)",
                        value=0,
                        scale=1,
                    )
                
                with gr.Row():
                    ds_shuffle = gr.Checkbox(label="Shuffle", value=True)
                    ds_load_btn = gr.Button("📥 Load Dataset", variant="primary")
                
                gr.Markdown("### Suggested Datasets")
                gr.Markdown(
                    "| Dataset | Subset | Good for |\n"
                    "|---------|--------|----------|\n"
                    "| `Anthropic/model-written-evals` | `persona` | Self-concept, identity |\n"
                    "| `Anthropic/model-written-evals` | `sycophancy` | Deference vs. honesty |\n"
                    "| `Anthropic/model-written-evals` | `advanced-ai-risk` | Self-preservation |\n"
                    "| `lmsys/chatbot_arena_conversations` | — | General conversation |\n"
                    "| `HuggingFaceH4/ultrachat_200k` | — | Diverse dialogue |\n"
                )
            
            # === TREE TAB ===
            with gr.Tab("Tree"):
                tree_display = gr.Code(
                    label="State Tree",
                    language=None,
                    interactive=False,
                    lines=15,
                )
                
                with gr.Row():
                    branch_name = gr.Textbox(label="Branch Name", scale=2)
                    branch_btn = gr.Button("🌿 Create Branch", scale=1)
                
                with gr.Row():
                    node_dropdown = gr.Dropdown(
                        label="Checkout Node",
                        choices=[],
                        interactive=True,
                        scale=2,
                    )
                    checkout_btn = gr.Button("📍 Checkout", scale=1)
                    refresh_btn = gr.Button("🔄", scale=0)
            
            # === COMPARE TAB ===
            with gr.Tab("Compare"):
                gr.Markdown("Compare outputs from two different branches on the same prompt.")
                
                compare_prompt = gr.Textbox(
                    label="Test Prompt",
                    lines=2,
                    placeholder="Enter a prompt to test both branches..."
                )
                
                with gr.Row():
                    compare_node_a = gr.Dropdown(label="Branch A", choices=[])
                    compare_node_b = gr.Dropdown(label="Branch B", choices=[])
                
                compare_btn = gr.Button("🔀 Compare")
                
                with gr.Row():
                    compare_out_a = gr.Markdown("**Branch A output:**")
                    compare_out_b = gr.Markdown("**Branch B output:**")
            
            # === SAVE TAB ===
            with gr.Tab("Save"):
                save_name = gr.Textbox(label="Adapter Name", value="my_trellis_adapter")
                save_btn = gr.Button("💾 Save Adapter")
            
            # === JOURNAL TAB ===
            with gr.Tab("Journal"):
                gr.Markdown(
                    "**Session Log:** Everything that happened this session (branches, checkouts, all choices).\n\n"
                    "**Lineage Narrative:** The clean story of just the current branch — good for blog posts."
                )
                
                with gr.Row():
                    refresh_log_btn = gr.Button("🔄 Refresh")
                
                with gr.Tabs():
                    with gr.Tab("Session Log"):
                        session_log_display = gr.Markdown(
                            value="*Load model to start logging.*",
                            label="Session Log",
                        )
                    
                    with gr.Tab("Lineage Narrative"):
                        lineage_display = gr.Markdown(
                            value="*No lineage yet.*",
                            label="Lineage",
                        )
                        
                        with gr.Row():
                            export_filename = gr.Textbox(
                                label="Export filename (optional)", 
                                placeholder="lineage_my_branch.md"
                            )
                            export_btn = gr.Button("📝 Export to File")
        
        # === WIRING ===
        
        def on_load():
            status, tree = app.initialize()
            choices = app.get_node_choices()
            return (
                status, 
                tree, 
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )
        
        load_btn.click(
            on_load,
            outputs=[status_box, tree_display, node_dropdown, compare_node_a, compare_node_b],
        )
        
        generate_btn.click(
            app.generate,
            inputs=[prompt_input],
            outputs=[opt_a, opt_b, opt_c, opt_d, status_box],
        )
        
        retaste_btn.click(
            app.retaste,
            outputs=[opt_a, opt_b, opt_c, opt_d, status_box],
        )
        
        # Dataset wiring
        def on_load_dataset(ds_id, subset, split, column, max_prompts, shuffle):
            status, ds_status = app.load_dataset(ds_id, subset, split, column, shuffle, max_prompts)
            return status, ds_status
        
        ds_load_btn.click(
            on_load_dataset,
            inputs=[ds_id_input, ds_subset_input, ds_split_input, ds_column_input, ds_max_input, ds_shuffle],
            outputs=[status_box, dataset_status],
        )
        
        def on_next_prompt():
            prompt, ds_status = app.next_prompt()
            return prompt, ds_status
        
        next_prompt_btn.click(
            on_next_prompt,
            outputs=[prompt_input, dataset_status],
        )
        
        def on_skip_prompt():
            prompt, ds_status = app.skip_prompt()
            return prompt, ds_status
        
        skip_prompt_btn.click(
            on_skip_prompt,
            outputs=[prompt_input, dataset_status],
        )
        
        def on_train(choice):
            status, tree, drift = app.train(choice)
            choices = app.get_node_choices()
            return (
                status, 
                tree, 
                drift,
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )
        
        train_btn.click(
            on_train,
            inputs=[choice_radio],
            outputs=[status_box, tree_display, drift_box, node_dropdown, compare_node_a, compare_node_b],
        )
        
        branch_btn.click(
            app.branch,
            inputs=[branch_name],
            outputs=[status_box, tree_display],
        )
        
        def on_checkout(node_id):
            status, tree, drift = app.checkout(node_id)
            return status, tree, drift
        
        checkout_btn.click(
            on_checkout,
            inputs=[node_dropdown],
            outputs=[status_box, tree_display, drift_box],
        )
        
        def refresh_dropdowns():
            choices = app.get_node_choices()
            return (
                gr.update(choices=choices),
                gr.update(choices=choices),
                gr.update(choices=choices),
            )
        
        refresh_btn.click(
            refresh_dropdowns,
            outputs=[node_dropdown, compare_node_a, compare_node_b],
        )
        
        compare_btn.click(
            app.compare,
            inputs=[compare_node_a, compare_node_b, compare_prompt],
            outputs=[compare_out_a, compare_out_b],
        )
        
        save_btn.click(
            app.save_current,
            inputs=[save_name],
            outputs=[status_box],
        )
        
        # Journal tab wiring
        def refresh_journal():
            return app.get_session_log(), app.get_lineage_narrative()
        
        refresh_log_btn.click(
            refresh_journal,
            outputs=[session_log_display, lineage_display],
        )
        
        export_btn.click(
            app.export_lineage,
            inputs=[export_filename],
            outputs=[status_box],
        )
    
    return demo


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Trellis: Interactive Preference Steering")
    parser.add_argument("--dry-run", action="store_true", help="Run UI without GPU")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    args = parser.parse_args()
    
    if args.config:
        config = TrellisConfig.load(args.config)
    else:
        config = TrellisConfig()
    
    app = TrellisApp(config, dry_run=args.dry_run)
    demo = build_ui(app)
    demo.launch()


if __name__ == "__main__":
    main()