"""
Review Screen (Screen 3)
========================

The final screen for reviewing training, exporting checkpoints, and merging LoRA.
"""

from __future__ import annotations

import gradio as gr


def build_review_screen() -> tuple[gr.Tab, dict]:
    """
    Build Screen 3: Review/Finalize.

    Returns:
        (tab, components_dict) for event wiring
    """
    with gr.Tab("Review", id=2) as tab:
        gr.Markdown("# Review & Export")
        gr.Markdown("Review your training session and export your trained model.")

        # ========== Stats Placeholder ==========
        with gr.Group():
            gr.Markdown("### Training Summary")
            with gr.Row():
                total_steps = gr.Markdown("**Total Steps:** 0")
                final_drift = gr.Markdown("**Final Drift:** 0.000")
            stats_placeholder = gr.Markdown(
                "*More detailed stats coming in a future update.*"
            )

        gr.Markdown("---")

        # ========== Config Review Placeholder ==========
        with gr.Accordion("Configuration Used", open=False):
            config_display = gr.JSON(label="Configuration")

        gr.Markdown("---")

        # ========== Journal Display ==========
        with gr.Group():
            gr.Markdown("### Training Journal")
            refresh_journal_btn = gr.Button("Refresh", size="sm")
            journal_display = gr.Markdown(
                "*Journal will be displayed here*",
                elem_classes=["journal-display"],
            )

        gr.Markdown("---")

        # ========== Export Options ==========
        with gr.Group():
            gr.Markdown("### Export Checkpoint")
            checkpoint_name = gr.Textbox(
                label="Checkpoint Name",
                placeholder="my_trellis_adapter",
                value="trellis_adapter",
            )
            save_checkpoint_btn = gr.Button(
                "Save LoRA Checkpoint",
                variant="secondary",
            )
            checkpoint_status = gr.Markdown("")

        gr.Markdown("---")

        # ========== Merge LoRA ==========
        with gr.Group():
            gr.Markdown("### Merge LoRA into Base Model")
            gr.Markdown(
                "*Optional: Merge the LoRA adapter into the base model to create "
                "a standalone model that can be shared without the base model.*"
            )
            gr.Markdown(
                "**Warning:** This temporarily requires significant additional memory.",
                elem_classes=["disk-warning"],
            )
            merge_path = gr.Textbox(
                label="Output Path",
                placeholder="./merged_model",
                value="./merged_model",
            )
            merge_btn = gr.Button("Merge & Save Full Model", variant="secondary")
            merge_status = gr.Markdown("")

        gr.Markdown("---")

        # ========== Start Over ==========
        with gr.Row():
            start_over_btn = gr.Button(
                "Start Over",
                variant="secondary",
                size="lg",
            )

    components = {
        # Stats
        "total_steps": total_steps,
        "final_drift": final_drift,
        "stats_placeholder": stats_placeholder,
        # Config
        "config_display": config_display,
        # Journal
        "refresh_journal_btn": refresh_journal_btn,
        "journal_display": journal_display,
        # Checkpoint
        "checkpoint_name": checkpoint_name,
        "save_checkpoint_btn": save_checkpoint_btn,
        "checkpoint_status": checkpoint_status,
        # Merge
        "merge_path": merge_path,
        "merge_btn": merge_btn,
        "merge_status": merge_status,
        # Navigation
        "start_over_btn": start_over_btn,
    }

    return tab, components
