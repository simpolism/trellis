"""
Config Screen (Screen 1)
========================

The opening configuration screen with dataset selection, model settings,
training parameters, and VRAM checking.
"""

from __future__ import annotations

import gradio as gr


def build_config_screen() -> tuple[gr.Tab, dict]:
    """
    Build Screen 1: Config/Opening.

    Returns:
        (tab, components_dict) for event wiring
    """
    with gr.Tab("Setup", id=0) as tab:
        gr.Markdown("# Trellis Setup")
        gr.Markdown("Configure your training session and start steering.")

        # ========== Session Resume Section ==========
        with gr.Group():
            gr.Markdown("### Quick Resume")
            with gr.Row():
                session_dropdown = gr.Dropdown(
                    label="Existing Sessions",
                    choices=[],
                    interactive=True,
                    scale=3,
                )
                refresh_sessions_btn = gr.Button("Refresh", scale=1, size="sm")
            resume_btn = gr.Button("Resume Session", variant="secondary")
            resume_status = gr.Markdown("")

        gr.Markdown("---")

        # ========== Dataset Section ==========
        with gr.Group():
            gr.Markdown("### Dataset")
            with gr.Row():
                dataset_input = gr.Textbox(
                    label="Dataset ID",
                    value="cosmicoptima/introspection-prompts",
                    placeholder="e.g., cosmicoptima/introspection-prompts",
                    scale=3,
                )
                dataset_subset = gr.Textbox(
                    label="Subset (optional)",
                    placeholder="e.g., persona",
                    scale=1,
                )
            with gr.Row():
                dataset_split = gr.Dropdown(
                    label="Split",
                    choices=["train", "test", "validation"],
                    value="train",
                    scale=1,
                )
                dataset_column = gr.Textbox(
                    label="Text Column (auto-detected)",
                    placeholder="Leave blank for auto",
                    scale=2,
                )
            load_preview_btn = gr.Button("Load & Preview", variant="secondary")
            dataset_status = gr.Markdown("")

            gr.Markdown("**Preview:**")
            preview_q1 = gr.Textbox(label="Question 1", interactive=False, lines=2)
            preview_q2 = gr.Textbox(label="Question 2", interactive=False, lines=2)
            preview_q3 = gr.Textbox(label="Question 3", interactive=False, lines=2)

        gr.Markdown("---")

        # ========== Model Section ==========
        with gr.Group():
            gr.Markdown("### Model")
            model_input = gr.Textbox(
                label="Model Name",
                value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                placeholder="HuggingFace model ID",
            )
            with gr.Row():
                context_slider = gr.Slider(
                    minimum=512,
                    maximum=8192,
                    value=2048,
                    step=256,
                    label="Context Length",
                    scale=2,
                )
                group_size = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Options per Prompt",
                    scale=1,
                )

        gr.Markdown("---")

        # ========== Training Engine Section ==========
        with gr.Group():
            gr.Markdown("### Training Engine")
            engine_dropdown = gr.Dropdown(
                choices=["UnSloth (4-bit + LoRA)"],
                value="UnSloth (4-bit + LoRA)",
                label="Engine",
                interactive=True,
            )

        gr.Markdown("---")

        # ========== Hyperparameters Section ==========
        with gr.Group():
            gr.Markdown("### Hyperparameters")
            with gr.Row():
                learning_rate = gr.Number(
                    value=2e-5,
                    label="Learning Rate",
                    precision=6,
                )
                kl_beta = gr.Number(
                    value=0.03,
                    label="KL Beta (anchor strength)",
                    precision=3,
                )
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Temperature",
                )
                max_new_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=256,
                    step=64,
                    label="Max New Tokens",
                )

        gr.Markdown("---")

        # ========== LoRA Settings Section ==========
        with gr.Group():
            gr.Markdown("### LoRA Settings")
            with gr.Row():
                lora_rank = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=16,
                    step=8,
                    label="Rank",
                    scale=1,
                )
                lora_alpha = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=16,
                    step=8,
                    label="Alpha",
                    scale=1,
                )
            max_undos = gr.Number(
                value=None,
                label="Max Undos to Save (blank = unlimited)",
                precision=0,
            )
            disk_warning = gr.Markdown(
                "*Note: Each checkpoint uses ~500MB for an 8B model. "
                "Leaving this blank will save all steps (recommended for short sessions).*",
                elem_classes=["disk-warning"],
            )

        gr.Markdown("---")

        # ========== System Prompt Section ==========
        with gr.Accordion("System Prompt & Wrapping", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt (optional)",
                placeholder="Custom system prompt for the model...",
                lines=3,
            )
            with gr.Row():
                prompt_prefix = gr.Textbox(
                    label="Prompt Prefix",
                    placeholder="Added before each prompt",
                    scale=1,
                )
                prompt_suffix = gr.Textbox(
                    label="Prompt Suffix",
                    placeholder="Added after each prompt",
                    scale=1,
                )

        gr.Markdown("---")

        # ========== VRAM Check Section ==========
        with gr.Group():
            gr.Markdown("### System Check")
            check_vram_btn = gr.Button("Check VRAM Requirements", variant="secondary")
            vram_display = gr.Markdown("*Click to check VRAM requirements*")

        gr.Markdown("---")

        # ========== Go Button ==========
        go_btn = gr.Button(
            "Go! Start Training",
            variant="primary",
            size="lg",
        )
        go_status = gr.Markdown("")

    components = {
        # Session resume
        "session_dropdown": session_dropdown,
        "refresh_sessions_btn": refresh_sessions_btn,
        "resume_btn": resume_btn,
        "resume_status": resume_status,
        # Dataset
        "dataset_input": dataset_input,
        "dataset_subset": dataset_subset,
        "dataset_split": dataset_split,
        "dataset_column": dataset_column,
        "load_preview_btn": load_preview_btn,
        "dataset_status": dataset_status,
        "preview_q1": preview_q1,
        "preview_q2": preview_q2,
        "preview_q3": preview_q3,
        # Model
        "model_input": model_input,
        "context_slider": context_slider,
        "group_size": group_size,
        # Engine
        "engine_dropdown": engine_dropdown,
        # Hyperparams
        "learning_rate": learning_rate,
        "kl_beta": kl_beta,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        # LoRA
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_undos": max_undos,
        "disk_warning": disk_warning,
        # Prompts
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_suffix": prompt_suffix,
        # VRAM
        "check_vram_btn": check_vram_btn,
        "vram_display": vram_display,
        # Go
        "go_btn": go_btn,
        "go_status": go_status,
    }

    return tab, components
