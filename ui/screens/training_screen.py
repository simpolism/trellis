"""
Training Screen (Screen 2)
==========================

The main training/game screen with option cards, undo, and real-time logging.
"""

from __future__ import annotations

import gradio as gr


def build_training_screen() -> tuple[gr.Tab, dict]:
    """
    Build Screen 2: Training/Game.

    Returns:
        (tab, components_dict) for event wiring
    """
    with gr.Tab("Train", id=1) as tab:
        # ========== Header Stats ==========
        with gr.Row(elem_classes=["stats-header"]):
            step_display = gr.Markdown("**Step:** 0")
            drift_display = gr.Markdown("**Drift:** 0.000")
            dataset_info = gr.Markdown("**Dataset:** Not loaded")

        gr.Markdown("---")

        # ========== Prompt Display ==========
        prompt_display = gr.Markdown(
            "*Waiting for prompt...*",
            elem_classes=["prompt-card"],
        )

        # ========== Edit Prompt ==========
        with gr.Accordion("Edit Question", open=False) as edit_accordion:
            edit_prompt_input = gr.Textbox(
                label="Edit the current question",
                lines=3,
                placeholder="Modify the question and regenerate...",
            )
            with gr.Row():
                apply_edit_btn = gr.Button("Apply & Regenerate", variant="secondary")
                cancel_edit_btn = gr.Button("Cancel", variant="stop")

        gr.Markdown("---")

        # ========== Option Cards ==========
        gr.Markdown("### Select your preference:")

        # Dynamic option display - supports up to 8 options
        option_btns = []
        with gr.Row(elem_classes=["option-row"]):
            opt_a = gr.Button("Option A", elem_classes=["option-card"], visible=True)
            opt_b = gr.Button("Option B", elem_classes=["option-card"], visible=True)
            option_btns.extend([opt_a, opt_b])

        with gr.Row(elem_classes=["option-row"]):
            opt_c = gr.Button("Option C", elem_classes=["option-card"], visible=True)
            opt_d = gr.Button("Option D", elem_classes=["option-card"], visible=True)
            option_btns.extend([opt_c, opt_d])

        # Extra options for group_size > 4
        with gr.Row(elem_classes=["option-row"], visible=False) as extra_row:
            opt_e = gr.Button("Option E", elem_classes=["option-card"], visible=False)
            opt_f = gr.Button("Option F", elem_classes=["option-card"], visible=False)
            opt_g = gr.Button("Option G", elem_classes=["option-card"], visible=False)
            opt_h = gr.Button("Option H", elem_classes=["option-card"], visible=False)
            option_btns.extend([opt_e, opt_f, opt_g, opt_h])

        gr.Markdown("---")

        # ========== Action Buttons ==========
        with gr.Row(elem_classes=["action-row"]):
            none_btn = gr.Button("None of these", variant="stop")
            skip_btn = gr.Button("Skip")
            undo_btn = gr.Button("Undo")

        undo_status = gr.Markdown("")

        gr.Markdown("---")

        # ========== Log Panel ==========
        with gr.Accordion("Log", open=True):
            log_output = gr.Textbox(
                lines=6,
                interactive=False,
                show_label=False,
                elem_classes=["log-panel"],
                placeholder="Training log will appear here...",
            )

        gr.Markdown("---")

        # ========== Bottom Actions ==========
        with gr.Row():
            save_session_btn = gr.Button("Save Session", variant="secondary")
            sample_btn = gr.Button("Sample", interactive=False)  # Placeholder
            journal_btn = gr.Button("View Journal", interactive=False)  # Placeholder
            done_btn = gr.Button("Done", variant="primary")

        save_status = gr.Markdown("")

    components = {
        # Header
        "step_display": step_display,
        "drift_display": drift_display,
        "dataset_info": dataset_info,
        # Prompt
        "prompt_display": prompt_display,
        # Edit
        "edit_accordion": edit_accordion,
        "edit_prompt_input": edit_prompt_input,
        "apply_edit_btn": apply_edit_btn,
        "cancel_edit_btn": cancel_edit_btn,
        # Options
        "opt_a": opt_a,
        "opt_b": opt_b,
        "opt_c": opt_c,
        "opt_d": opt_d,
        "opt_e": opt_e,
        "opt_f": opt_f,
        "opt_g": opt_g,
        "opt_h": opt_h,
        "option_btns": option_btns,
        "extra_row": extra_row,
        # Actions
        "none_btn": none_btn,
        "skip_btn": skip_btn,
        "undo_btn": undo_btn,
        "undo_status": undo_status,
        # Log
        "log_output": log_output,
        # Bottom
        "save_session_btn": save_session_btn,
        "sample_btn": sample_btn,
        "journal_btn": journal_btn,
        "done_btn": done_btn,
        "save_status": save_status,
    }

    return tab, components
