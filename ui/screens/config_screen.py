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
            gr.Markdown(
                "*Continue a previous training session. Sessions are automatically saved "
                "after each preference selection, so you can resume right where you left off.*"
            )
            with gr.Row():
                session_dropdown = gr.Dropdown(
                    label="Existing Sessions",
                    choices=[],
                    interactive=True,
                    scale=2,
                    info="Select a saved session to resume",
                )
                refresh_sessions_btn = gr.Button("🔄", scale=0, size="sm", min_width=40)
            resume_btn = gr.Button("Resume Session Now", variant="primary")
            resume_status = gr.Markdown("")

        gr.Markdown("---")

        # ========== Dataset Section ==========
        with gr.Group():
            gr.Markdown("### Dataset")
            gr.Markdown(
                "*Choose the prompts that will be used to train your model. "
                "These questions help shape the model's personality and responses.*"
            )
            with gr.Row():
                dataset_input = gr.Textbox(
                    label="Dataset ID",
                    value="cosmicoptima/introspection-prompts",
                    placeholder="e.g., cosmicoptima/introspection-prompts",
                    scale=3,
                    info="HuggingFace dataset ID (e.g., 'username/dataset') or local folder path",
                )
                dataset_subset = gr.Textbox(
                    label="Subset (optional)",
                    placeholder="e.g., persona",
                    scale=1,
                    info="Some datasets have multiple subsets",
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
                    info="Column containing prompts (usually auto-detected)",
                )
            load_preview_btn = gr.Button("Load & Preview Dataset", variant="secondary")
            dataset_status = gr.Markdown("")

            gr.Markdown("**Preview:**")
            preview_q1 = gr.Textbox(label="Question 1", interactive=False, lines=2)
            preview_q2 = gr.Textbox(label="Question 2", interactive=False, lines=2)
            preview_q3 = gr.Textbox(label="Question 3", interactive=False, lines=2)

        gr.Markdown("---")

        # ========== Model Section ==========
        with gr.Group():
            gr.Markdown("### Model")
            gr.Markdown(
                "*Select the base model to train. The model will be loaded in 4-bit "
                "quantization with LoRA adapters for efficient training.*"
            )
            model_input = gr.Textbox(
                label="Model Name",
                value="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                placeholder="HuggingFace model ID or local path",
                info="HuggingFace model ID (e.g., 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit') or local folder path",
            )
            with gr.Row():
                context_slider = gr.Slider(
                    minimum=512,
                    maximum=8192,
                    value=2048,
                    step=256,
                    label="Context Length",
                    scale=2,
                    info="Max tokens per prompt+response. Higher = more memory",
                )
                group_size = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1,
                    label="Options per Prompt",
                    scale=1,
                    info="How many response options to generate each turn",
                )

            with gr.Row():
                check_vram_btn = gr.Button("Check VRAM", variant="secondary", scale=1)
                load_model_btn = gr.Button("Load Model", variant="secondary", scale=1)

            vram_display = gr.Markdown("*Click 'Check VRAM' to estimate memory requirements*")
            model_status = gr.Markdown("")

        gr.Markdown("---")

        # ========== Training Engine Section ==========
        with gr.Group():
            gr.Markdown("### Training Engine")
            gr.Markdown(
                "*The training backend handles model loading and gradient updates. "
                "UnSloth provides optimized 4-bit quantization with LoRA.*"
            )
            engine_dropdown = gr.Dropdown(
                choices=["UnSloth (4-bit + LoRA)"],
                value="UnSloth (4-bit + LoRA)",
                label="Engine",
                interactive=True,
            )

            # Engine-specific settings (generation)
            gr.Markdown("#### Generation Settings")
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.2,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative/random. Lower = more focused/deterministic",
                )
                max_new_tokens = gr.Slider(
                    minimum=64,
                    maximum=1024,
                    value=256,
                    step=64,
                    label="Max New Tokens",
                    info="Maximum length of generated responses",
                )

            # Engine-specific settings (training)
            gr.Markdown("#### Training Settings")
            with gr.Row():
                learning_rate = gr.Number(
                    value=2e-5,
                    label="Learning Rate",
                    precision=6,
                    info="How fast the model learns. Default (2e-5) is conservative",
                )
                kl_beta = gr.Number(
                    value=0.03,
                    label="KL Beta",
                    precision=3,
                    info="Anchor strength. Higher = stays closer to base model",
                )

            # LoRA settings
            gr.Markdown("#### LoRA Settings")
            gr.Markdown(
                "*LoRA (Low-Rank Adaptation) trains small adapter layers instead of the full model, "
                "dramatically reducing memory usage.*"
            )
            with gr.Row():
                lora_rank = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=16,
                    step=8,
                    label="Rank",
                    scale=1,
                    info="Adapter capacity. Higher = more expressive but uses more memory",
                )
                lora_alpha = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=16,
                    step=8,
                    label="Alpha",
                    scale=1,
                    info="Scaling factor. Usually set equal to rank",
                )

        gr.Markdown("---")

        # ========== Undo & Checkpoints Section ==========
        with gr.Group():
            gr.Markdown("### Undo & Checkpoints")
            gr.Markdown(
                "*Trellis saves checkpoints after each step, allowing you to undo mistakes. "
                "Limit the number of saved checkpoints to save disk space.*"
            )
            max_undos = gr.Number(
                value=None,
                label="Max Checkpoints to Keep",
                precision=0,
                info="Leave blank for unlimited (recommended for short sessions)",
            )
            disk_warning = gr.Markdown(
                "*Each checkpoint uses ~500MB for an 8B model. "
                "Unlimited checkpoints recommended for sessions under 50 steps.*",
            )

        gr.Markdown("---")

        # ========== System Prompt Section ==========
        with gr.Accordion("System Prompt & Wrapping (Advanced)", open=False):
            gr.Markdown(
                "*Customize how prompts are formatted before being sent to the model. "
                "Most users can leave these blank.*"
            )
            system_prompt = gr.Textbox(
                label="System Prompt (optional)",
                placeholder="Custom system prompt for the model...",
                lines=3,
                info="Sets the model's role or persona",
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
        "check_vram_btn": check_vram_btn,
        "load_model_btn": load_model_btn,
        "vram_display": vram_display,
        "model_status": model_status,
        # Engine
        "engine_dropdown": engine_dropdown,
        # Training (now in engine section)
        "learning_rate": learning_rate,
        "kl_beta": kl_beta,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        # LoRA (now in engine section)
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_undos": max_undos,
        "disk_warning": disk_warning,
        # Prompts
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_suffix": prompt_suffix,
        # Go
        "go_btn": go_btn,
        "go_status": go_status,
    }

    return tab, components
