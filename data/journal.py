"""
Session Journal
===============

Dual-mode logging for Trellis sessions.
- Session log: Chronological record of everything
- Lineage narrative: Clean story of choices for writeups
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from state.checkpoint import Checkpoint


class Journal:
    """
    Dual-mode logging for Trellis sessions.

    - Session log: Chronological record of everything (trains, skips, undos)
    - Lineage narrative: Clean story from start to current state (for writeups)
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

    def _timestamp(self) -> str:
        """Get current timestamp for log entries."""
        return datetime.now().strftime("%H:%M:%S")

    def log_init(self, model_name: str):
        """Log model initialization."""
        self._append(f"[{self._timestamp()}] Loaded model: `{model_name}`\n")

    def log_generation(self, prompt: str, num_options: int):
        """Log a generation event (simple 1-line format)."""
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        self._append(f"[{self._timestamp()}] Generated {num_options} options for: {prompt_preview}")

    def log_train(
        self,
        step: int,
        prompt: str,
        choice: str,
        chosen_text: str,
        drift: float,
        metrics: dict,
    ):
        """Log a training step (simple format for real-time display)."""
        loss = metrics.get("total_loss", 0)
        self._append(f"[{self._timestamp()}] Step {step}: {choice} | drift={drift:.3f} | loss={loss:.4f}")

    def log_train_detailed(
        self,
        step: int,
        prompt: str,
        choice: str,
        chosen_text: str,
        drift: float,
        metrics: dict,
    ):
        """Log a detailed training step (for full journal display)."""
        entry = f"""### Step {step}
**Prompt:**
> {prompt}

**Choice:** {choice} | **Drift:** {drift:.3f} | **Loss:** {metrics.get('total_loss', 0):.4f}

<details>
<summary>Selected response</summary>

{chosen_text}

</details>

"""
        self._append(entry)

    def log_skip(self, step: int, prompt: str):
        """Log a skipped prompt."""
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        self._append(f"[{self._timestamp()}] Skipped: {prompt_preview}")

    def log_undo(self, from_step: int, to_step: int):
        """Log an undo action."""
        self._append(f"[{self._timestamp()}] Undo: step {from_step} -> step {to_step}")

    def log_reject_all(self, step: int, prompt: str, drift: float):
        """Log a reject-all training step."""
        self._append(f"[{self._timestamp()}] Step {step}: Reject All | drift={drift:.3f}")

    def log_session_save(self, path: str):
        """Log session save."""
        self._append(f"\n[{self._timestamp()}] Session saved to: {path}\n")

    def get_content(self) -> str:
        """Read and return the full session log content."""
        if self.session_path.exists():
            return self.session_path.read_text()
        return "*No log yet*"

    def generate_lineage_narrative(self, checkpoints: list["Checkpoint"]) -> str:
        """
        Generate a clean narrative from the checkpoint history.
        This is the "story" of choices, suitable for blog posts.
        """
        if not checkpoints:
            return "*No history yet.*"

        current = checkpoints[-1] if checkpoints else None
        title = f"Step {current.step_count}" if current else "unknown"

        lines = [
            f"# Training Lineage: {title}",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "A narrative of the choices that shaped this model.",
            "",
            "---",
            "",
        ]

        # Skip root (step 0, no choices made)
        training_steps = [c for c in checkpoints if c.step_count > 0 and c.choice]

        if not training_steps:
            lines.append("*No training steps yet — this is the base model.*")
        else:
            for i, checkpoint in enumerate(training_steps, 1):
                prompt = checkpoint.prompt or ""
                prompt_preview = prompt[:300] + "..." if len(prompt) > 300 else prompt

                lines.append(f"## Step {i}")
                lines.append("")

                # Prompt
                if checkpoint.prompt:
                    lines.append("**Prompt:**")
                    lines.append(f"> {prompt_preview}")
                    lines.append("")

                # Choice label
                lines.append(f"**Selected:** {checkpoint.choice}")
                lines.append("")

                # The actual response that was chosen
                if checkpoint.chosen_response:
                    response_preview = checkpoint.chosen_response[:600]
                    if len(checkpoint.chosen_response) > 600:
                        response_preview += "..."
                    lines.append("**Response:**")
                    lines.append("")
                    for response_line in response_preview.split('\n'):
                        lines.append(f"> {response_line}")
                    lines.append("")
                elif checkpoint.choice == "Reject All":
                    lines.append("*(All options rejected)*")
                    lines.append("")

                lines.append(f"*Drift from base: {checkpoint.drift_from_base:.3f}*")
                lines.append("")
                lines.append("---")
                lines.append("")

        # Summary
        if current:
            lines.extend([
                "## Summary",
                "",
                f"- **Total steps:** {current.step_count}",
                f"- **Final drift:** {current.drift_from_base:.3f}",
            ])

        return "\n".join(lines)

    def save_lineage(
        self,
        checkpoints: list["Checkpoint"],
        filename: Optional[str] = None
    ) -> Path:
        """Save lineage narrative to a file."""
        narrative = self.generate_lineage_narrative(checkpoints)

        if filename is None:
            step = checkpoints[-1].step_count if checkpoints else 0
            filename = f"lineage_step_{step}.md"

        path = self.save_dir / filename
        with open(path, "w") as f:
            f.write(narrative)

        return path
