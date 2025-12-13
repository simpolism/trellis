# 🌿 Trellis

Interactive preference steering with branching state. Shape a model's personality through direct vibe-checks, explore divergent directions, and keep a narrative of how each branch came to be.

## What it does

1. **Prompt** → model generates 4 response variants
2. **Pick** → select the one with the right vibe (or reject all)
3. **Update** → LoRA weights shift toward your preference
4. **Branch** → explore alternatives without losing prior work

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
python trellis.py           # launch UI
python trellis.py --dry-run # UI without GPU (for testing)
```

## Hardware

Designed for ~16GB VRAM (e.g., RTX 4070 Ti Super). Uses 4-bit quantized base + LoRA.

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

## Tabs

| Tab | Purpose |
|-----|---------|
| **Steer** | Core loop: generate, pick, train, re-taste |
| **Dataset** | Load HuggingFace datasets for prompts |
| **Tree** | View state tree, branch, checkout previous states |
| **Compare** | A/B test two branches on the same prompt |
| **Save** | Export adapter to disk |
| **Journal** | Session log (everything) + lineage narrative (current branch's story) |

## Config

Edit `TrellisConfig` in `trellis.py` or pass `--config path/to/config.json`:

```python
model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
lora_rank: int = 16
group_size: int = 4          # variants per generation
temperature: float = 1.2     # sampling diversity  
learning_rate: float = 2e-5
kl_beta: float = 0.03        # anchor to base policy (0 disables)
```

## License

MIT