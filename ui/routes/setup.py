"""
Setup Routes
============

Routes for the setup/configuration screen.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Form, Cookie, Response
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from ui.session_manager import session_manager

router = APIRouter(prefix="/setup", tags=["setup"])


@router.post("/preview-dataset")
async def preview_dataset(
    dataset_id: str = Form(...),
    dataset_subset: str = Form(None),
    dataset_split: str = Form("train"),
    dataset_column: str = Form("text"),
    session_id: str = Cookie(None),
):
    """Load dataset and return preview HTML fragment."""
    app = session_manager.get_app(session_id)
    if not app:
        return "<p><strong>Error:</strong> No session found</p>"

    try:
        status, q1, q2, q3 = app.preview_dataset(
            dataset_id=dataset_id,
            subset=dataset_subset if dataset_subset else None,
            split=dataset_split,
            column=dataset_column if dataset_column else None,
        )

        # Return HTML fragment
        html = f"""
        <div style="margin-top: 16px;">
            <p><strong>Status:</strong> {status}</p>
            <hr>
            <h4>Preview Questions:</h4>
            <div class="form-group">
                <label>Question 1</label>
                <textarea readonly rows="2">{q1}</textarea>
            </div>
            <div class="form-group">
                <label>Question 2</label>
                <textarea readonly rows="2">{q2}</textarea>
            </div>
            <div class="form-group">
                <label>Question 3</label>
                <textarea readonly rows="2">{q3}</textarea>
            </div>
        </div>
        """
        return HTMLResponse(content=html)

    except Exception as e:
        return HTMLResponse(content=f"<p><strong>Error:</strong> {str(e)}</p>")


@router.post("/load-model")
async def load_model(
    model_name: str = Form(...),
    precision: str = Form("4-bit"),
    session_id: str = Cookie(None),
):
    """Load model with SSE status streaming."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p><strong>Error:</strong> No session found</p>")

    async def event_stream() -> AsyncIterator[dict]:
        """Stream model loading status."""
        try:
            # Use default parameters for a simple load
            load_in_4bit = "4-bit" in precision.lower()

            for status in app.load_model_only(
                model_name=model_name,
                context_length=4096,
                group_size=4,
                engine_name="UnSloth (LoRA)",
                learning_rate=2e-5,
                kl_beta=0.03,
                temperature=1.2,
                max_new_tokens=256,
                lora_rank=16,
                lora_alpha=16,
                max_undos=None,
                system_prompt="",
                prompt_prefix="",
                prompt_suffix="",
                dataset_id="abhayesian/introspection-prompts",
                precision_choice=precision,
                append_think_tag=True,
            ):
                yield {"event": "message", "data": status}

            # Send completion event
            yield {"event": "message", "data": "✅ Model loaded successfully!"}
            yield {"event": "complete", "data": ""}

        except Exception as e:
            yield {"event": "error", "data": f"❌ Error: {str(e)}"}
            yield {"event": "complete", "data": ""}

    return EventSourceResponse(event_stream())


@router.post("/resume-session")
async def resume_session(
    session_select: str = Form(...),
    session_id: str = Cookie(None),
):
    """Resume an existing session with SSE status streaming."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p><strong>Error:</strong> No session found</p>")

    if not session_select:
        return HTMLResponse(content="<p><strong>Error:</strong> Please select a session</p>")

    async def event_stream() -> AsyncIterator[dict]:
        """Stream session resume status."""
        try:
            for status in app.resume_session(session_select):
                yield {"event": "message", "data": status}

            # Send redirect on success
            yield {"event": "message", "data": "✅ Session restored!"}
            yield {"event": "redirect", "data": "/training"}
            yield {"event": "complete", "data": ""}

        except Exception as e:
            yield {"event": "error", "data": f"❌ Error: {str(e)}"}
            yield {"event": "complete", "data": ""}

    return EventSourceResponse(event_stream())


@router.post("/start-training")
async def start_training(
    model_name: str = Form(...),
    context_length: int = Form(4096),
    group_size: int = Form(4),
    engine_name: str = Form("UnSloth (LoRA)"),
    learning_rate: float = Form(2e-5),
    kl_beta: float = Form(0.03),
    temperature: float = Form(1.2),
    max_new_tokens: int = Form(256),
    lora_rank: int = Form(16),
    lora_alpha: int = Form(16),
    max_undos: int = Form(None),
    system_prompt: str = Form(""),
    prompt_prefix: str = Form(""),
    prompt_suffix: str = Form(""),
    dataset_id: str = Form(...),
    dataset_subset: str = Form(None),
    dataset_split: str = Form("train"),
    dataset_column: str = Form("text"),
    precision: str = Form("4-bit"),
    append_think_tag: bool = Form(True),
    session_id: str = Cookie(None),
):
    """Start training session with SSE status streaming."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p><strong>Error:</strong> No session found</p>")

    async def event_stream() -> AsyncIterator[dict]:
        """Stream training initialization status."""
        try:
            for status in app.start_training(
                model_name=model_name,
                context_length=context_length,
                group_size=group_size,
                engine_name=engine_name,
                learning_rate=learning_rate,
                kl_beta=kl_beta,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                max_undos=max_undos,
                system_prompt=system_prompt if system_prompt else "",
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                dataset_id=dataset_id,
                dataset_subset=dataset_subset if dataset_subset else None,
                dataset_split=dataset_split,
                dataset_column=dataset_column if dataset_column else None,
                precision_choice=precision,
                append_think_tag=append_think_tag,
            ):
                yield {"event": "message", "data": status}

                # Check if we're done
                if status == "Ready!":
                    yield {"event": "redirect", "data": "/training"}
                    yield {"event": "complete", "data": ""}
                    return

        except Exception as e:
            yield {"event": "error", "data": f"❌ Error: {str(e)}"}
            yield {"event": "complete", "data": ""}

    return EventSourceResponse(event_stream())
