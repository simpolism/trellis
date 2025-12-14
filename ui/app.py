"""
Trellis App Controller
======================

Main controller that wires together the engine, state, and UI components.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TrellisConfig
from data.prompt_source import PromptSource
from data.journal import Journal
from engine.base import BaseEngine, VRAMEstimate
from engine.unsloth_engine import UnslothEngine
from engine.vram import estimate_vram_unsloth
from state.checkpoint import Checkpoint
from state.undo_stack import LinearUndoStack
from state.session import SessionState, load_session

from .styles import MOBILE_CSS
from .screens import build_config_screen, build_training_screen, build_review_screen


class TrellisApp:
    """Main controller wiring engine, state, and Gradio UI."""

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

    # =========================================================================
    # Screen 1: Config Methods
    # =========================================================================

    def discover_sessions(self) -> list[tuple[str, str]]:
        """Find existing sessions for resume dropdown."""
        sessions = SessionState.discover_sessions(str(self.base_save_dir))
        result = []
        for dir_name, s, path in sessions:
            # Use custom name if set, otherwise use directory name
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
        status = self.prompt_source.load(
            dataset_id=dataset_id,
            subset=subset if subset else None,
            split=split,
            text_column=column if column else None,
            shuffle=True,
        )

        preview = self.prompt_source.preview(3)
        q1 = preview[0] if len(preview) > 0 else ""
        q2 = preview[1] if len(preview) > 1 else ""
        q3 = preview[2] if len(preview) > 2 else ""

        return status, q1, q2, q3

    def check_vram(
        self,
        model_name: str,
        context_length: int,
        group_size: int,
        lora_rank: int,
    ) -> str:
        """Estimate VRAM requirements."""
        temp_config = TrellisConfig(
            model_name=model_name,
            max_seq_length=int(context_length),
            group_size=int(group_size),
            lora_rank=int(lora_rank),
        )
        estimate = estimate_vram_unsloth(temp_config)
        return estimate.to_display_string()

    def start_training(
        self,
        # Model
        model_name: str,
        context_length: int,
        group_size: int,
        # Engine (currently ignored, only one engine)
        engine_name: str,
        # Hyperparams
        learning_rate: float,
        kl_beta: float,
        temperature: float,
        max_new_tokens: int,
        # LoRA
        lora_rank: int,
        lora_alpha: int,
        max_undos: Optional[float],
        # Prompts
        system_prompt: str,
        prompt_prefix: str,
        prompt_suffix: str,
        # Dataset (already loaded via preview)
        dataset_id: str,
    ):
        """
        Initialize training session.

        Yields status updates for streaming UI.
        """
        yield "Initializing session..."

        # Create session directory
        session_name = SessionState.generate_session_name()
        self.session_dir = self.base_save_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

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
            save_dir=str(self.session_dir),
            default_dataset=dataset_id,
            system_prompt=system_prompt if system_prompt else None,
            prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix,
        )

        # Save config
        self.config.save(str(self.session_dir / "config.json"))

        # Initialize session state
        self.session_state = SessionState.create_new(self.session_dir, self.config)

        # Initialize journal
        self.journal = Journal(self.session_dir / "journal")

        # Initialize undo stack
        self.undo_stack = LinearUndoStack(
            self.session_dir,
            max_undos=self.config.max_undos,
        )

        yield "Loading model (this may take a minute)..."

        # Initialize engine
        self.engine = UnslothEngine(self.config)
        status = self.engine.load_model()

        self.journal.log_init(self.config.model_name)

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
        print(f"[Trellis] Session started")
        print(f"[Trellis] Model: {self.config.model_name}")
        print(f"[Trellis] Dataset: {self.prompt_source.dataset_id} ({self.prompt_source.total()} prompts)")

        yield "Ready!"

    def resume_session(self, session_path: str):
        """Resume an existing session."""
        yield "Loading session..."

        self.session_dir = Path(session_path)

        # Load session state and config
        self.session_state, self.config = load_session(session_path)

        yield "Restoring undo stack..."

        # Load undo stack
        self.undo_stack = LinearUndoStack.load(self.session_dir)

        yield "Loading model..."

        # Initialize engine
        self.engine = UnslothEngine(self.config)
        self.engine.load_model()

        yield "Restoring checkpoint..."

        # Restore to current checkpoint
        current = self.undo_stack.current()
        if current and current.step_count > 0:
            adapter_state, optimizer_state = current.load_tensors()
            self.engine.restore_state(adapter_state, optimizer_state)

        # Restore prompt source
        if self.session_state.prompt_source_dataset_id:
            self.prompt_source.restore_state({
                "dataset_id": self.session_state.prompt_source_dataset_id,
                "subset": self.session_state.prompt_source_subset,
                "split": self.session_state.prompt_source_split,
                "index": self.session_state.prompt_source_index,
            })

        # Restore journal (just open existing)
        self.journal = Journal(self.session_dir / "journal")

        self.step_count = current.step_count if current else 0

        print(f"[Trellis] Resumed session at step {self.step_count}")

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

        print(f"[Trellis] Generating options for: {self.current_prompt[:80]}...")

        # Generate options
        self.current_options = self.engine.generate_options(self.current_prompt)

        print(f"[Trellis] Generated {len(self.current_options)} options")

        if self.journal:
            self.journal.log_generation(self.current_prompt, len(self.current_options))

        return self.current_prompt, self.current_options

    def select_option(self, choice_idx: int) -> tuple[str, dict]:
        """Select an option and train."""
        if not self.engine or not self.current_options:
            return "No options to select from", {}

        choice_label = f"Option {chr(65 + choice_idx)}" if choice_idx < len(self.current_options) else "Reject All"
        chosen_response = self.current_options[choice_idx] if choice_idx < len(self.current_options) else ""

        print(f"[Trellis] Training on {choice_label}...")

        # Train
        status, metrics = self.engine.train_step(choice_idx)
        drift = self.engine.compute_drift()

        self.step_count += 1

        print(f"[Trellis] Step {self.step_count}: {status} | drift={drift:.3f}")

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
                self.prompt_source.index,
            )
            self.session_state.save(self.session_dir / "session.json")

        return status, {"step": self.step_count, "drift": drift, **metrics}

    def skip_prompt(self) -> str:
        """Skip current prompt without training."""
        if self.journal and self.current_prompt:
            self.journal.log_skip(self.step_count, self.current_prompt)

        print(f"[Trellis] Skipped prompt")

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

            print(f"[Trellis] Undo: step {old_step} -> {self.step_count}")

            return checkpoint, f"Undone to step {self.step_count}"

        return None, "Undo failed"

    def apply_edited_prompt(self, new_prompt: str) -> tuple[str, list[str]]:
        """Apply an edited prompt and regenerate."""
        self.current_prompt = new_prompt

        print(f"[Trellis] Regenerating with edited prompt...")

        self.current_options = self.engine.generate_options(new_prompt)

        print(f"[Trellis] Generated {len(self.current_options)} options")

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

        print(f"[Trellis] Session saved: {name or self.session_dir.name}")

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

        return f"LoRA checkpoint saved to {path}"

    def merge_lora(self, output_path: str) -> str:
        """Merge LoRA into base model."""
        if not self.engine:
            return "No model loaded"

        return self.engine.merge_and_save(output_path)

    def get_final_stats(self) -> tuple[str, str]:
        """Get final stats for review screen."""
        steps_text = f"**Total Steps:** {self.step_count}"
        drift = self.engine.compute_drift() if self.engine else 0.0
        drift_text = f"**Final Drift:** {drift:.3f}"
        return steps_text, drift_text

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _timestamp(self) -> str:
        """Get current timestamp for logs."""
        return datetime.now().strftime("%H:%M:%S")


# JavaScript for scrolling to top of page on tab switch
SCROLL_TO_TOP_JS = """
function scrollToTop() {
    window.scrollTo({top: 0, behavior: 'smooth'});
    return [];
}
"""


def build_ui(app: TrellisApp) -> gr.Blocks:
    """Build the complete Gradio UI and wire up events."""

    with gr.Blocks(title="Trellis", css=MOBILE_CSS, js=SCROLL_TO_TOP_JS) as demo:
        gr.Markdown("# Trellis")
        gr.Markdown("*Interactive preference steering for language models*")

        with gr.Tabs() as tabs:
            # Build screens
            config_tab, config_components = build_config_screen()
            training_tab, training_components = build_training_screen()
            review_tab, review_components = build_review_screen()

        # =====================================================================
        # Screen 1 Event Handlers
        # =====================================================================

        def refresh_sessions():
            sessions = app.discover_sessions()
            choices = [s[0] for s in sessions]
            return gr.Dropdown(choices=choices)

        config_components["refresh_sessions_btn"].click(
            refresh_sessions,
            outputs=[config_components["session_dropdown"]],
        )

        def preview_dataset(dataset_id, subset, split, column):
            status, q1, q2, q3 = app.preview_dataset(dataset_id, subset, split, column)
            return status, q1, q2, q3

        config_components["load_preview_btn"].click(
            preview_dataset,
            inputs=[
                config_components["dataset_input"],
                config_components["dataset_subset"],
                config_components["dataset_split"],
                config_components["dataset_column"],
            ],
            outputs=[
                config_components["dataset_status"],
                config_components["preview_q1"],
                config_components["preview_q2"],
                config_components["preview_q3"],
            ],
        )

        def check_vram(model, context, group, rank):
            return app.check_vram(model, context, group, rank)

        config_components["check_vram_btn"].click(
            check_vram,
            inputs=[
                config_components["model_input"],
                config_components["context_slider"],
                config_components["group_size"],
                config_components["lora_rank"],
            ],
            outputs=[config_components["vram_display"]],
        )

        def format_options_for_display(options: list[str]) -> list[str]:
            """Format options as markdown for display (no truncation)."""
            formatted = []
            for i, opt in enumerate(options[:8]):
                label = chr(65 + i)
                formatted.append(f"**Option {label}:**\n\n{opt}")
            return formatted

        def start_training_flow(
            model, context, group, engine,
            lr, kl, temp, tokens,
            rank, alpha, max_undos,
            sys_prompt, prefix, suffix,
            dataset_id,
        ):
            """Start training with streaming updates, then generate first prompt."""
            # Yield status updates during loading
            for status in app.start_training(
                model, context, group, engine,
                lr, kl, temp, tokens,
                rank, alpha, max_undos,
                sys_prompt, prefix, suffix,
                dataset_id,
            ):
                yield (
                    status,  # go_status
                    gr.Tabs(selected=0),  # stay on setup tab during loading
                    gr.skip(), gr.skip(), gr.skip(),  # stats
                    gr.skip(),  # prompt
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),  # options A-D
                )

            # Generate first prompt
            prompt, options = app.generate_and_display()
            stats = app.get_stats()

            # Format options for display
            opt_texts = format_options_for_display(options)

            # Switch to training tab
            yield (
                "Ready!",  # go_status
                gr.Tabs(selected=1),  # switch to training tab
                stats[0], stats[1], stats[2],  # stats
                f"**Prompt:**\n\n{prompt}",  # prompt display
                opt_texts[0] if len(opt_texts) > 0 else "*Option A*",
                opt_texts[1] if len(opt_texts) > 1 else "*Option B*",
                opt_texts[2] if len(opt_texts) > 2 else "*Option C*",
                opt_texts[3] if len(opt_texts) > 3 else "*Option D*",
            )

        config_components["go_btn"].click(
            start_training_flow,
            inputs=[
                config_components["model_input"],
                config_components["context_slider"],
                config_components["group_size"],
                config_components["engine_dropdown"],
                config_components["learning_rate"],
                config_components["kl_beta"],
                config_components["temperature"],
                config_components["max_new_tokens"],
                config_components["lora_rank"],
                config_components["lora_alpha"],
                config_components["max_undos"],
                config_components["system_prompt"],
                config_components["prompt_prefix"],
                config_components["prompt_suffix"],
                config_components["dataset_input"],
            ],
            outputs=[
                config_components["go_status"],
                tabs,
                training_components["step_display"],
                training_components["drift_display"],
                training_components["dataset_info"],
                training_components["prompt_display"],
                training_components["opt_a_text"],
                training_components["opt_b_text"],
                training_components["opt_c_text"],
                training_components["opt_d_text"],
            ],
        )

        # Resume session handler
        def resume_session_flow(session_dropdown):
            """Resume session and go directly to training."""
            if not session_dropdown:
                yield "Please select a session to resume", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
                return

            # Find the session path from the dropdown value
            sessions = app.discover_sessions()
            session_path = None
            for display_name, path in sessions:
                if display_name == session_dropdown:
                    session_path = path
                    break

            if not session_path:
                yield "Session not found", gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
                return

            # Resume session
            for status in app.resume_session(session_path):
                yield (
                    status,
                    gr.Tabs(selected=0),
                    gr.skip(), gr.skip(), gr.skip(),
                    gr.skip(),
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                )

            # Generate first prompt
            prompt, options = app.generate_and_display()
            stats = app.get_stats()
            opt_texts = format_options_for_display(options)

            yield (
                "Resumed!",
                gr.Tabs(selected=1),
                stats[0], stats[1], stats[2],
                f"**Prompt:**\n\n{prompt}",
                opt_texts[0] if len(opt_texts) > 0 else "*Option A*",
                opt_texts[1] if len(opt_texts) > 1 else "*Option B*",
                opt_texts[2] if len(opt_texts) > 2 else "*Option C*",
                opt_texts[3] if len(opt_texts) > 3 else "*Option D*",
            )

        config_components["resume_btn"].click(
            resume_session_flow,
            inputs=[config_components["session_dropdown"]],
            outputs=[
                config_components["resume_status"],
                tabs,
                training_components["step_display"],
                training_components["drift_display"],
                training_components["dataset_info"],
                training_components["prompt_display"],
                training_components["opt_a_text"],
                training_components["opt_b_text"],
                training_components["opt_c_text"],
                training_components["opt_d_text"],
            ],
        )

        # =====================================================================
        # Screen 2 Event Handlers
        # =====================================================================

        # Common outputs for option/action buttons
        training_outputs = [
            training_components["prompt_display"],
            training_components["step_display"],
            training_components["drift_display"],
            training_components["dataset_info"],
            training_components["opt_a_text"],
            training_components["opt_b_text"],
            training_components["opt_c_text"],
            training_components["opt_d_text"],
            training_components["undo_status"],
        ]

        def generate_next_with_loading():
            """Generate next prompt and options."""
            prompt, options = app.generate_and_display()
            stats = app.get_stats()
            opt_texts = format_options_for_display(options)

            return (
                f"**Prompt:**\n\n{prompt}",
                stats[0], stats[1], stats[2],
                opt_texts[0] if len(opt_texts) > 0 else "*Option A*",
                opt_texts[1] if len(opt_texts) > 1 else "*Option B*",
                opt_texts[2] if len(opt_texts) > 2 else "*Option C*",
                opt_texts[3] if len(opt_texts) > 3 else "*Option D*",
                "",  # Clear undo status
            )

        def select_and_advance(choice_idx):
            """Select option, train, then generate next."""
            app.select_option(choice_idx)
            return generate_next_with_loading()

        # Wire option select buttons
        training_components["opt_a_btn"].click(
            lambda: select_and_advance(0),
            outputs=training_outputs,
        )
        training_components["opt_b_btn"].click(
            lambda: select_and_advance(1),
            outputs=training_outputs,
        )
        training_components["opt_c_btn"].click(
            lambda: select_and_advance(2),
            outputs=training_outputs,
        )
        training_components["opt_d_btn"].click(
            lambda: select_and_advance(3),
            outputs=training_outputs,
        )

        def reject_all():
            """Reject all options and train."""
            app.select_option(app.config.group_size if app.config else 4)
            return generate_next_with_loading()

        training_components["none_btn"].click(
            reject_all,
            outputs=training_outputs,
        )

        def skip():
            app.skip_prompt()
            return generate_next_with_loading()

        training_components["skip_btn"].click(
            skip,
            outputs=training_outputs,
        )

        def undo():
            checkpoint, status = app.undo()
            if checkpoint and checkpoint.options:
                # Restore previous options
                prompt = checkpoint.prompt or "*Previous prompt*"
                options = checkpoint.options
                opt_texts = format_options_for_display(options)
                stats = app.get_stats()

                return (
                    f"**Prompt:**\n\n{prompt}",
                    stats[0], stats[1], stats[2],
                    opt_texts[0] if len(opt_texts) > 0 else "*Option A*",
                    opt_texts[1] if len(opt_texts) > 1 else "*Option B*",
                    opt_texts[2] if len(opt_texts) > 2 else "*Option C*",
                    opt_texts[3] if len(opt_texts) > 3 else "*Option D*",
                    status,
                )
            else:
                return (
                    gr.skip(),
                    gr.skip(), gr.skip(), gr.skip(),
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    status,
                )

        training_components["undo_btn"].click(
            undo,
            outputs=training_outputs,
        )

        # Inline prompt editing handlers
        def enter_edit_mode():
            """Switch to edit mode."""
            return (
                gr.Markdown(visible=False),  # hide prompt display
                gr.Textbox(visible=True, value=app.current_prompt or ""),  # show edit input
                gr.Row(visible=True),  # show edit buttons
                gr.Button(visible=False),  # hide edit button
            )

        def cancel_edit_mode():
            """Exit edit mode without changes."""
            return (
                gr.Markdown(visible=True),  # show prompt display
                gr.Textbox(visible=False),  # hide edit input
                gr.Row(visible=False),  # hide edit buttons
                gr.Button(visible=True),  # show edit button
            )

        def apply_edit_and_regenerate(new_prompt):
            """Apply edited prompt and regenerate options."""
            prompt, options = app.apply_edited_prompt(new_prompt)
            opt_texts = format_options_for_display(options)
            stats = app.get_stats()

            return (
                gr.Markdown(visible=True, value=f"**Prompt:**\n\n{prompt}"),  # show updated prompt
                gr.Textbox(visible=False),  # hide edit input
                gr.Row(visible=False),  # hide edit buttons
                gr.Button(visible=True),  # show edit button
                opt_texts[0] if len(opt_texts) > 0 else "*Option A*",
                opt_texts[1] if len(opt_texts) > 1 else "*Option B*",
                opt_texts[2] if len(opt_texts) > 2 else "*Option C*",
                opt_texts[3] if len(opt_texts) > 3 else "*Option D*",
            )

        training_components["edit_prompt_btn"].click(
            enter_edit_mode,
            outputs=[
                training_components["prompt_display"],
                training_components["prompt_edit_input"],
                training_components["edit_buttons_row"],
                training_components["edit_prompt_btn"],
            ],
        )

        training_components["cancel_edit_btn"].click(
            cancel_edit_mode,
            outputs=[
                training_components["prompt_display"],
                training_components["prompt_edit_input"],
                training_components["edit_buttons_row"],
                training_components["edit_prompt_btn"],
            ],
        )

        training_components["apply_edit_btn"].click(
            apply_edit_and_regenerate,
            inputs=[training_components["prompt_edit_input"]],
            outputs=[
                training_components["prompt_display"],
                training_components["prompt_edit_input"],
                training_components["edit_buttons_row"],
                training_components["edit_prompt_btn"],
                training_components["opt_a_text"],
                training_components["opt_b_text"],
                training_components["opt_c_text"],
                training_components["opt_d_text"],
            ],
        )

        # Save session with name
        def save_session(name):
            return app.save_session_with_name(name)

        training_components["save_session_btn"].click(
            save_session,
            inputs=[training_components["save_session_name"]],
            outputs=[training_components["save_status"]],
        )

        # Go to review
        def go_to_review():
            stats = app.get_final_stats()
            journal = app.get_journal_content()
            config = app.get_config_display()
            return gr.Tabs(selected=2), stats[0], stats[1], journal, config

        training_components["done_btn"].click(
            go_to_review,
            outputs=[
                tabs,
                review_components["total_steps"],
                review_components["final_drift"],
                review_components["journal_display"],
                review_components["config_display"],
            ],
        )

        # =====================================================================
        # Screen 3 Event Handlers
        # =====================================================================

        def save_checkpoint(name):
            return app.save_checkpoint(name)

        review_components["save_checkpoint_btn"].click(
            save_checkpoint,
            inputs=[review_components["checkpoint_name"]],
            outputs=[review_components["checkpoint_status"]],
        )

        def merge_model(path):
            # Show loading message first
            yield "Merging model... This may take several minutes."
            result = app.merge_lora(path)
            yield result

        review_components["merge_btn"].click(
            merge_model,
            inputs=[review_components["merge_path"]],
            outputs=[review_components["merge_status"]],
        )

        def start_over():
            sessions = app.discover_sessions()
            choices = [s[0] for s in sessions]
            return gr.Tabs(selected=0), gr.Dropdown(choices=choices)

        review_components["start_over_btn"].click(
            start_over,
            outputs=[tabs, config_components["session_dropdown"]],
        )

        # =====================================================================
        # Initial load
        # =====================================================================

        def on_load():
            sessions = app.discover_sessions()
            choices = [s[0] for s in sessions]
            return gr.Dropdown(choices=choices)

        demo.load(on_load, outputs=[config_components["session_dropdown"]])

    return demo
