"""
Trellis App Controller
======================

Main controller that wires together the engine, state, and UI components.
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TrellisConfig
from data.prompt_source import PromptSource
from data.journal import Journal
from engine.base import BaseEngine
from engine.unsloth_engine import UnslothEngine
from state.checkpoint import Checkpoint
from state.undo_stack import LinearUndoStack
from state.session import SessionState, load_session


def _log(message: str):
    """Print a timestamped log message to stdout."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [Trellis] {message}")


class TrellisApp:
    """Main controller wiring engine, state, and UI."""

    def __init__(self, base_save_dir: str = "./trellis_sessions"):
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        # Will be initialized on start
        self.config: Optional[TrellisConfig] = None
        self.engine: Optional[BaseEngine] = None
        self.undo_stack: Optional[LinearUndoStack] = None
        self.prompt_source: PromptSource = PromptSource()
        self.journal: Optional[Journal] = None
        self.session_state: Optional[SessionState] = None
        self.session_dir: Optional[Path] = None

        # Training state
        self.step_count: int = 0
        self.current_prompt: Optional[str] = None
        self.current_options: list[str] = []

        # Model loading state (separate from training)
        self.model_loaded: bool = False

    def _create_engine(self, engine_name: str) -> BaseEngine:
        """Factory for selecting the training backend."""
        normalized = (engine_name or "").lower()
        if "unsloth" in normalized:
            return UnslothEngine(self.config)
        raise ValueError(f"Unsupported engine: {engine_name}")

    # =========================================================================
    # Screen 1: Config Methods
    # =========================================================================

    def discover_sessions(self) -> list[tuple[str, str]]:
        """Find existing sessions for resume dropdown."""
        sessions = SessionState.discover_sessions(str(self.base_save_dir))
        result = []
        for dir_name, s, path in sessions:
            display_name = s.name if s.name else dir_name
            date_str = s.last_modified[:10] if s.last_modified else "unknown"
            result.append((f"{display_name} ({date_str})", path))
        return result

    def preview_dataset(
        self,
        dataset_id: str,
        subset: str,
        split: str,
        column: str,
    ) -> tuple[str, str, str, str]:
        """Load dataset and return preview questions."""
        _log(f"Loading dataset: {dataset_id}")
        status = self.prompt_source.load(
            dataset_id=dataset_id,
            subset=subset if subset else None,
            split=split,
            text_column=column if column else None,
            shuffle=True,
        )
        _log(f"Dataset loaded: {self.prompt_source.total()} prompts")

        preview = self.prompt_source.preview(3)
        q1 = preview[0] if len(preview) > 0 else ""
        q2 = preview[1] if len(preview) > 1 else ""
        q3 = preview[2] if len(preview) > 2 else ""

        return status, q1, q2, q3

    def load_model_only(
        self,
        model_name: str,
        context_length: int,
        group_size: int,
        engine_name: str,
        learning_rate: float,
        kl_beta: float,
        temperature: float,
        max_new_tokens: int,
        lora_rank: int,
        lora_alpha: int,
        max_undos: Optional[float],
        system_prompt: str,
        prompt_prefix: str,
        prompt_suffix: str,
        dataset_id: str,
        precision_choice: str,
        append_think_tag: bool,
    ):
        """Load model without starting training session. Yields status updates."""
        _log(f"Loading model: {model_name}")
        yield "Loading model (this may take a minute)..."

        load_in_4bit = "4-bit" in (precision_choice or "").lower()

        # Build config
        self.config = TrellisConfig(
            model_name=model_name,
            max_seq_length=int(context_length),
            group_size=int(group_size),
            learning_rate=learning_rate,
            kl_beta=kl_beta,
            temperature=temperature,
            max_new_tokens=int(max_new_tokens),
            lora_rank=int(lora_rank),
            lora_alpha=int(lora_alpha),
            max_undos=int(max_undos) if max_undos else None,
            default_dataset=dataset_id,
            system_prompt=system_prompt if system_prompt else None,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            load_in_4bit=load_in_4bit,
            append_think_tag=append_think_tag,
        )

        # Initialize engine
        self.engine = self._create_engine(engine_name)
        status = self.engine.load_model()
        self.model_loaded = True

        _log(f"Model loaded successfully")
        yield f"✅ Model loaded: {model_name}"

    def start_training(
        self,
        model_name: str,
        context_length: int,
        group_size: int,
        engine_name: str,
        learning_rate: float,
        kl_beta: float,
        temperature: float,
        max_new_tokens: int,
        lora_rank: int,
        lora_alpha: int,
        max_undos: Optional[float],
        system_prompt: str,
        prompt_prefix: str,
        prompt_suffix: str,
        dataset_id: str,
        dataset_subset: str,
        dataset_split: str,
        dataset_column: str,
        precision_choice: str,
        append_think_tag: bool,
    ):
        """Initialize training session. Yields status updates."""
        _log("Initializing training session...")
        yield "Initializing session..."

        load_in_4bit = "4-bit" in (precision_choice or "").lower()

        # Create session directory
        session_name = SessionState.generate_session_name()
        self.session_dir = self.base_save_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Build/update config with save_dir
        self.config = TrellisConfig(
            model_name=model_name,
            max_seq_length=int(context_length),
            group_size=int(group_size),
            learning_rate=learning_rate,
            kl_beta=kl_beta,
            temperature=temperature,
            max_new_tokens=int(max_new_tokens),
            lora_rank=int(lora_rank),
            lora_alpha=int(lora_alpha),
            max_undos=int(max_undos) if max_undos else None,
            save_dir=str(self.session_dir),
            default_dataset=dataset_id,
            system_prompt=system_prompt if system_prompt else None,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
            load_in_4bit=load_in_4bit,
            append_think_tag=append_think_tag,
        )

        # Save config
        self.config.save(str(self.session_dir / "config.json"))

        # Initialize session state
        self.session_state = SessionState.create_new(
            self.session_dir,
            self.config,
            engine_name=engine_name,
        )

        # Initialize journal
        self.journal = Journal(self.session_dir / "journal")

        # Initialize undo stack
        self.undo_stack = LinearUndoStack(
            self.session_dir,
            max_undos=self.config.max_undos,
        )

        # Load dataset (ensures prompt source is ready before first generate)
        dataset_status = self.prompt_source.load(
            dataset_id=dataset_id,
            subset=dataset_subset if dataset_subset else None,
            split=dataset_split or "train",
            text_column=dataset_column if dataset_column else None,
            shuffle=True,
        )
        _log(f"Dataset load: {dataset_status}")
        lower_status = dataset_status.lower()
        if (
            lower_status.startswith("failed")
            or "not found" in lower_status
            or "install datasets" in lower_status
            or self.prompt_source.total() == 0
        ):
            yield f"Dataset error: {dataset_status}"
            return

        # Persist dataset info to session metadata
        if self.session_state:
            self.session_state.update_prompt_source(
                self.prompt_source.dataset_id,
                self.prompt_source.subset,
                self.prompt_source.split,
                self.prompt_source.text_column,
                self.prompt_source.index,
            )
            self.session_state.save(self.session_dir / "session.json")

        # Load model if not already loaded
        if (
            not self.model_loaded
            or self.engine is None
            or self.engine.name != engine_name
        ):
            _log(f"Loading model: {model_name}")
            yield "Loading model (this may take a minute)..."
            self.engine = self._create_engine(engine_name)
            self.engine.load_model()
            self.model_loaded = True

        self.journal.log_init(self.config.model_name)

        _log("Creating initial checkpoint...")
        yield "Creating initial checkpoint..."

        # Create root checkpoint
        root = Checkpoint.create(
            step_count=0,
            adapter_state=self.engine.get_adapter_state(),
            optimizer_state=self.engine.get_optimizer_state(),
            drift_from_base=0.0,
        )
        self.undo_stack.push(root)

        self.step_count = 0
        _log("Session initialized")
        _log(f"Model: {self.config.model_name}")
        _log(f"Dataset: {self.prompt_source.dataset_id} ({self.prompt_source.total()} prompts)")

        yield "Ready!"

    def resume_session(self, session_path: str):
        """Resume an existing session. Yields status updates."""
        _log(f"Resuming session: {session_path}")
        yield "Loading session..."

        self.session_dir = Path(session_path)

        # Load session state and config
        self.session_state, self.config = load_session(session_path)
        engine_name = self.session_state.engine_name or "UnSloth (LoRA)"

        _log("Restoring undo stack...")
        yield "Restoring undo stack..."

        # Load undo stack
        self.undo_stack = LinearUndoStack.load(self.session_dir)

        _log(f"Loading model: {self.config.model_name}")
        yield "Loading model..."

        # Initialize engine
        self.engine = self._create_engine(engine_name)
        self.engine.load_model()
        self.model_loaded = True

        _log("Restoring checkpoint...")
        yield "Restoring checkpoint..."

        # Restore to current checkpoint
        current = self.undo_stack.current()
        if current and current.step_count > 0:
            adapter_state, optimizer_state = current.load_tensors()
            self.engine.restore_state(adapter_state, optimizer_state)

        # Restore prompt source
        if self.session_state.prompt_source_dataset_id:
            yield "Restoring dataset..."
            status = self.prompt_source.restore_state({
                "dataset_id": self.session_state.prompt_source_dataset_id,
                "subset": self.session_state.prompt_source_subset,
                "split": self.session_state.prompt_source_split,
                "text_column": self.session_state.prompt_source_text_column,
                "index": self.session_state.prompt_source_index,
            })
            _log(f"Dataset restore: {status}")
            lower_status = status.lower()
            if (
                lower_status.startswith("failed")
                or "not found" in lower_status
                or "install datasets" in lower_status
                or self.prompt_source.total() == 0
            ):
                yield f"Dataset error: {status}"
                return

        # Restore journal (just open existing)
        self.journal = Journal(self.session_dir / "journal")

        self.step_count = current.step_count if current else 0

        _log(f"Session resumed at step {self.step_count}")

        yield "Session restored!"

    # =========================================================================
    # Screen 2: Training Methods
    # =========================================================================

    def generate_and_display(self) -> tuple[str, list[str]]:
        """Get next prompt and generate options."""
        if not self.engine or not self.engine.is_loaded:
            return "Model not loaded", []

        # Get next prompt
        self.current_prompt = self.prompt_source.next()
        if not self.current_prompt:
            return "No prompts available. Load a dataset first.", []

        _log(f"Generating options for prompt: {self.current_prompt[:60]}...")

        # Generate options
        self.current_options = self.engine.generate_options(self.current_prompt)

        _log(f"Generated {len(self.current_options)} options")

        if self.journal:
            self.journal.log_generation(self.current_prompt, len(self.current_options))

        return self.current_prompt, self.current_options

    def select_option(self, choice_idx: int) -> tuple[str, dict]:
        """Select an option and train."""
        if not self.engine or not self.current_options:
            return "No options to select from", {}

        choice_label = f"Option {chr(65 + choice_idx)}" if choice_idx < len(self.current_options) else "Reject All"
        chosen_response = self.current_options[choice_idx] if choice_idx < len(self.current_options) else ""

        _log(f"Training on {choice_label}...")

        # Train
        status, metrics = self.engine.train_step(choice_idx)
        drift = self.engine.compute_drift()

        self.step_count += 1

        _log(f"Step {self.step_count}: {status} | drift={drift:.3f}")

        # Log to journal (detailed format)
        if self.journal:
            if choice_idx < len(self.current_options):
                self.journal.log_train_detailed(
                    self.step_count,
                    self.current_prompt or "",
                    choice_label,
                    chosen_response,
                    drift,
                    metrics,
                )
            else:
                self.journal.log_reject_all(
                    self.step_count,
                    self.current_prompt or "",
                    drift,
                )

        # Create checkpoint
        checkpoint = Checkpoint.create(
            step_count=self.step_count,
            prompt=self.current_prompt,
            choice=choice_label,
            chosen_response=chosen_response,
            options=self.current_options.copy(),
            drift_from_base=drift,
            adapter_state=self.engine.get_adapter_state(),
            optimizer_state=self.engine.get_optimizer_state(),
        )
        self.undo_stack.push(checkpoint)

        # Update session state
        if self.session_state:
            self.session_state.update_step(self.step_count)
            self.session_state.update_prompt_source(
                self.prompt_source.dataset_id,
                self.prompt_source.subset,
                self.prompt_source.split,
                self.prompt_source.text_column,
                self.prompt_source.index,
            )
            self.session_state.save(self.session_dir / "session.json")

        # Free previous batch to reclaim GPU memory between steps
        self.engine.current_batch = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return status, {"step": self.step_count, "drift": drift, **metrics}

    def skip_prompt(self) -> str:
        """Skip current prompt without training."""
        if self.journal and self.current_prompt:
            self.journal.log_skip(self.step_count, self.current_prompt)

        _log("Skipped prompt")

        return "Skipped"

    def undo(self) -> tuple[Optional[Checkpoint], str]:
        """Undo to previous checkpoint."""
        if not self.undo_stack or not self.undo_stack.can_undo():
            return None, "Nothing to undo"

        old_step = self.step_count
        checkpoint = self.undo_stack.undo()

        if checkpoint:
            # Restore engine state
            adapter_state, optimizer_state = checkpoint.load_tensors()
            self.engine.restore_state(adapter_state, optimizer_state)

            self.step_count = checkpoint.step_count
            self.current_prompt = checkpoint.prompt
            self.current_options = checkpoint.options

            if self.journal:
                self.journal.log_undo(old_step, self.step_count)

            _log(f"Undo: step {old_step} -> {self.step_count}")

            return checkpoint, f"Undone to step {self.step_count}"

        return None, "Undo failed"

    def apply_edited_prompt(self, new_prompt: str) -> tuple[str, list[str]]:
        """Apply an edited prompt and regenerate."""
        self.current_prompt = new_prompt

        _log("Regenerating with edited prompt...")

        self.current_options = self.engine.generate_options(new_prompt)

        _log(f"Generated {len(self.current_options)} options")

        return new_prompt, self.current_options

    def save_session_with_name(self, name: str) -> str:
        """Save current session with a custom name."""
        if not self.session_state or not self.session_dir:
            return "No active session"

        # Update session name if provided
        if name and name.strip():
            self.session_state.name = name.strip()

        self.session_state.save(self.session_dir / "session.json")

        if self.journal:
            self.journal.log_session_save(str(self.session_dir))

        _log(f"Session saved: {name or self.session_dir.name}")

        return f"Session saved as '{name or self.session_dir.name}'"

    def get_stats(self) -> tuple[str, str, str]:
        """Get current stats for display."""
        step_text = f"**Step:** {self.step_count}"
        drift = self.engine.compute_drift() if self.engine else 0.0
        drift_text = f"**Drift:** {drift:.3f}"
        dataset_text = f"**Dataset:** {self.prompt_source.status()}"
        return step_text, drift_text, dataset_text

    # =========================================================================
    # Screen 3: Review Methods
    # =========================================================================

    def get_journal_content(self) -> str:
        """Get full journal content for display."""
        if self.journal:
            return self.journal.get_content()
        return "*No journal available*"

    def get_config_display(self) -> dict:
        """Get config as dict for JSON display."""
        if self.config:
            return self.config.to_dict()
        return {}

    def save_checkpoint(self, name: str) -> str:
        """Save LoRA checkpoint."""
        if not self.engine:
            return "No model loaded"

        path = self.session_dir / name if self.session_dir else Path(name)
        self.engine.save_adapter(str(path))

        _log(f"LoRA checkpoint saved to {path}")

        return f"LoRA checkpoint saved to {path}"

    def merge_lora(self, output_path: str) -> str:
        """Merge LoRA into base model."""
        if not self.engine:
            return "No model loaded"

        _log(f"Merging LoRA to {output_path}...")
        result = self.engine.merge_and_save(output_path)
        _log(f"Merge complete: {result}")
        return result

    def unload_model(self):
        """Unload the model to free memory."""
        if self.engine:
            _log("Unloading model...")
            # Clear references to allow garbage collection
            self.engine = None
            self.model_loaded = False

            # Try to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

            _log("Model unloaded")

    def get_final_stats(self) -> tuple[str, str]:
        """Get final stats for review screen."""
        steps_text = f"**Total Steps:** {self.step_count}"
        drift = self.engine.compute_drift() if self.engine else 0.0
        drift_text = f"**Final Drift:** {drift:.3f}"
        return steps_text, drift_text
