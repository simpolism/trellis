# Repository Guidelines

## Project Structure & Module Organization
- `trellis.py` is the entry point that boots the Gradio UI; `config.py` holds the tunable `TrellisConfig` dataclass.
- `engine/` contains training/back-end logic (Unsloth adapter, VRAM helpers); `state/` tracks session checkpoints, undo stack, and tree metadata.
- `data/` manages prompt sources and journaling; `ui/` houses the Gradio app wiring plus screen/component layouts and shared styles.
- Runtime artifacts live under `trellis_sessions/` and `trellis_states/`; generated weights and compiled kernels land in `merged_model/` and `unsloth_compiled_cache/` (all gitignored—leave untouched or recreate).

## Build, Test, and Development Commands
- Create a local env (Python 3.12): `uv venv --python 3.12 .venv && source .venv/bin/activate`
- Install deps (CUDA toolchain required for Torch/Unsloth pins): `uv pip install -r requirements.txt`
- Launch full app: `python trellis.py`
- Isolate artifacts while testing: `python trellis.py --save-dir /tmp/trellis_sessions`
- Use the repo virtualenv for Python (`source .venv/bin/activate`); system `python` is not available
- Default model: `unsloth/gemma-3-1b-it-unsloth-bnb-4bit` (1B, 4096 ctx). `<think>` is auto-added to replies by default; toggle precision (16-bit vs 4-bit) and thinking in the Setup screen.
- AGENTS.md and CLAUDE.md are hard-linked; edits to one update the other

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indents and type hints; keep functions small and side-effect-light.
- Classes use `PascalCase`, functions/vars use `snake_case`, config keys mirror training params (`lora_rank`, `kl_beta`, etc.).
- Prefer dataclasses for structured config/state, and keep UI logic in `ui/` thin by delegating to app/state helpers.

## Testing Guidelines
- No automated test suite yet; rely on manual passes: `python trellis.py` for end-to-end checks with a small dataset.
- When adding tests, prefer `pytest`, keep GPU-heavy paths optional/mocked, and seed any sampling to stabilize outputs.

## Commit & Pull Request Guidelines
- Use short, imperative commits (e.g., `Next pass bug fixes`, `Full Refactor`); keep each commit scoped to one concern.
- PRs should summarize behavior changes, list the commands used to validate (dataset selections, precision choice), and link issues.
- For UI changes, attach a screenshot/GIF of the affected screens; for training changes, note dataset, adapter path, and VRAM footprint.

## Security & Configuration Tips
- Do not commit runtime artifacts (`trellis_sessions/`, `trellis_states/`, `merged_model/`, `unsloth_compiled_cache/`) or private tokens; use env vars for secrets.
- Large model/cache updates can bloat the repo—prefer documenting the download path instead of checking in binaries.
- If you tweak defaults in `config.py`, keep GPU requirements in mind and update README/CLI flags where relevant.
