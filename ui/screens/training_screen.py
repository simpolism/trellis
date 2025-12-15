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

        # ========== Prompt Display (Inline Editable) ==========
        # The prompt area can switch between display and edit modes
        prompt_display = gr.Markdown(
            "*Waiting for prompt...*",
            elem_classes=["prompt-card"],
            visible=True,
        )
        prompt_edit_input = gr.Textbox(
            label="Edit Prompt",
            lines=4,
            visible=False,
            placeholder="Edit the prompt and regenerate...",
        )
        with gr.Row(visible=False) as edit_buttons_row:
            apply_edit_btn = gr.Button("Apply & Regenerate", variant="primary", size="sm")
            cancel_edit_btn = gr.Button("Cancel", variant="secondary", size="sm")

        # Edit button (toggles prompt to edit mode)
        edit_prompt_btn = gr.Button("✏️ Edit Prompt", size="sm", variant="secondary")

        gr.Markdown("---")

        # ========== Option Cards ==========
        # Each option shows full text in a plain textbox with a select button
        gr.Markdown("### Select your preference:")

        # Option A
        with gr.Group(elem_classes=["option-group"]):
            gr.Markdown("**A**", elem_classes=["option-label"])
            opt_a_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_a_btn = gr.Button("Select A", size="sm", variant="primary", elem_classes=["select-btn"])

        # Option B
        with gr.Group(elem_classes=["option-group"]):
            gr.Markdown("**B**", elem_classes=["option-label"])
            opt_b_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_b_btn = gr.Button("Select B", size="sm", variant="primary", elem_classes=["select-btn"])

        # Option C
        with gr.Group(elem_classes=["option-group"]):
            gr.Markdown("**C**", elem_classes=["option-label"])
            opt_c_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_c_btn = gr.Button("Select C", size="sm", variant="primary", elem_classes=["select-btn"])

        # Option D
        with gr.Group(elem_classes=["option-group"]):
            gr.Markdown("**D**", elem_classes=["option-label"])
            opt_d_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_d_btn = gr.Button("Select D", size="sm", variant="primary", elem_classes=["select-btn"])

        # Extra options for group_size > 4 (hidden by default)
        with gr.Group(elem_classes=["option-group"], visible=False) as opt_e_group:
            gr.Markdown("**E**", elem_classes=["option-label"])
            opt_e_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_e_btn = gr.Button("Select E", size="sm", variant="primary", elem_classes=["select-btn"])

        with gr.Group(elem_classes=["option-group"], visible=False) as opt_f_group:
            gr.Markdown("**F**", elem_classes=["option-label"])
            opt_f_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_f_btn = gr.Button("Select F", size="sm", variant="primary", elem_classes=["select-btn"])

        with gr.Group(elem_classes=["option-group"], visible=False) as opt_g_group:
            gr.Markdown("**G**", elem_classes=["option-label"])
            opt_g_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_g_btn = gr.Button("Select G", size="sm", variant="primary", elem_classes=["select-btn"])

        with gr.Group(elem_classes=["option-group"], visible=False) as opt_h_group:
            gr.Markdown("**H**", elem_classes=["option-label"])
            opt_h_text = gr.Textbox(
                value="", show_label=False, interactive=False,
                lines=5, max_lines=10, elem_classes=["option-text"]
            )
            opt_h_btn = gr.Button("Select H", size="sm", variant="primary", elem_classes=["select-btn"])

        gr.Markdown("---")

        # ========== Action Buttons ==========
        with gr.Row(elem_classes=["action-row"]):
            none_btn = gr.Button("None of these", variant="stop")
            skip_btn = gr.Button("Skip")
            undo_btn = gr.Button("Undo")

        undo_status = gr.Markdown("", visible=True)

        gr.Markdown("---")

        # ========== Bottom Actions ==========
        with gr.Row():
            with gr.Column(scale=2):
                save_session_name = gr.Textbox(
                    label="Session Name",
                    placeholder="my_training_session",
                    info="Name for saving this session",
                )
            with gr.Column(scale=1):
                save_session_btn = gr.Button("Save Session", variant="secondary")

        save_status = gr.Markdown("")

        gr.Markdown("---")

        with gr.Row():
            sample_btn = gr.Button("Sample", interactive=False)
            journal_btn = gr.Button("View Journal", interactive=False)
            done_btn = gr.Button("Done - Review Results", variant="primary")

    components = {
        # Header
        "step_display": step_display,
        "drift_display": drift_display,
        "dataset_info": dataset_info,
        # Prompt (inline editable)
        "prompt_display": prompt_display,
        "prompt_edit_input": prompt_edit_input,
        "edit_buttons_row": edit_buttons_row,
        "apply_edit_btn": apply_edit_btn,
        "cancel_edit_btn": cancel_edit_btn,
        "edit_prompt_btn": edit_prompt_btn,
        # Options - text displays
        "opt_a_text": opt_a_text,
        "opt_b_text": opt_b_text,
        "opt_c_text": opt_c_text,
        "opt_d_text": opt_d_text,
        "opt_e_text": opt_e_text,
        "opt_f_text": opt_f_text,
        "opt_g_text": opt_g_text,
        "opt_h_text": opt_h_text,
        # Options - select buttons
        "opt_a_btn": opt_a_btn,
        "opt_b_btn": opt_b_btn,
        "opt_c_btn": opt_c_btn,
        "opt_d_btn": opt_d_btn,
        "opt_e_btn": opt_e_btn,
        "opt_f_btn": opt_f_btn,
        "opt_g_btn": opt_g_btn,
        "opt_h_btn": opt_h_btn,
        # Option groups (for visibility control)
        "opt_e_group": opt_e_group,
        "opt_f_group": opt_f_group,
        "opt_g_group": opt_g_group,
        "opt_h_group": opt_h_group,
        # Actions
        "none_btn": none_btn,
        "skip_btn": skip_btn,
        "undo_btn": undo_btn,
        "undo_status": undo_status,
        # Save
        "save_session_name": save_session_name,
        "save_session_btn": save_session_btn,
        "save_status": save_status,
        # Bottom (with future feature placeholders)
        "sample_btn": sample_btn,
        "journal_btn": journal_btn,
        "done_btn": done_btn,
    }

    return tab, components
