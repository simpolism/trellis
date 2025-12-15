# 🌿 Trellis

Interactive preference steering with linear undo. Shape a model's personality through direct vibe-checks, walk changes forward/backward with checkpoints, and keep a narrative of how each step was made. Default setup uses **PleIAs/Baguettotron** (321M) in 16-bit with a LoRA head, with a toggle to fall back to 4-bit if VRAM is tight.

## What it does

1. **Prompt** → model generates 4 response variants
2. **Pick** → select the one with the right vibe (or reject all)
3. **Update** → LoRA weights shift toward your preference
4. **Checkpoint** → linear undo stack captures adapter/optimizer state for rewind

The state tree lets you checkpoint, branch, and compare different personality trajectories. The journal logs everything for later writeups.

## Install

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

UV_TORCH_BACKEND=cu128 uv pip install torch==2.9.1
uv pip install "unsloth[cu128-torch291] @ git+https://github.com/unslothai/unsloth.git"
uv pip install gradio peft packaging ninja einops
# Flash attention: optional but better perf
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-linux_x86_64.whl
```


## Run

```bash
python trellis.py           # launch UI on port 7860
python trellis.py --share   # launch with public URL
python trellis.py --port 7861 --host 127.0.0.1
```

## Hardware

Default Baguettotron runs comfortably in 16-bit on midrange GPUs (the 321M base is small). Use the **Precision** toggle in Setup to switch to 4-bit quantization if you need to squeeze VRAM.

## Prompt format (Baguettotron)

The model uses a Qwen-style chat template:

```xml
<|im_start|>user
Who are you?<|im_end|>
<|im_start|>assistant
<think>
```

Trellis auto-appends `<think>` to the assistant preamble by default; disable it in **System Prompt & Wrapping** if you want to try replies without a thinking trace.
Max context length is 4096 tokens for this model; set the slider accordingly in Setup.

## Using Datasets

Load prompts directly from HuggingFace in the **Dataset** tab:

| Dataset | Subset | Good for |
|---------|--------|----------|
| `Anthropic/model-written-evals` | `persona` | Self-concept, identity |
| `Anthropic/model-written-evals` | `sycophancy` | Deference vs. honesty |
| `Anthropic/model-written-evals` | `advanced-ai-risk` | Self-preservation |
| `lmsys/chatbot_arena_conversations` | — | General conversation |
| `HuggingFaceH4/ultrachat_200k` | — | Diverse dialogue |

Once loaded, use **Next Prompt** in the Steer tab to pull prompts automatically.

## UI Flow

| Tab | Purpose |
|-----|---------|
| **Config** | Enter model + training hyperparams, dataset id/subset/column, VRAM estimate, and preview prompts |
| **Train** | Generate options, pick/reject, checkpoint each step, undo/redo, edit prompts inline |
| **Review** | Inspect session journal, view final stats, save adapters, merge LoRA, and start over |

## Config

Configure from the UI. Each session saves `config.json` alongside checkpoints. Key fields:

- Model: `model_name`, `max_seq_length`, `load_in_4bit`
- Generation: `group_size`, `temperature`, `max_new_tokens`, `min_p`
- Training: `learning_rate`, `lora_rank`, `lora_alpha`, `kl_beta`, `max_undos`
- Prompt shaping: `system_prompt`, `prompt_prefix`, `prompt_suffix`, `append_think_tag`
- Dataset source: HF dataset id/subset/split/column (auto-detected if possible)

Currently ships with the Unsloth engine; additional engines will be selectable in future releases.

## License

MIT
