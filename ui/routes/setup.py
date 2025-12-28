"""
Setup Routes
============

Routes for the setup/configuration screen.
"""

from __future__ import annotations

from typing import AsyncIterator, Optional

from fastapi import APIRouter, Form, Cookie, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from ui.session_manager import session_manager
from state.session import load_session

router = APIRouter(prefix="/setup", tags=["setup"])


@router.get("/session-config")
async def get_session_config(session_path: str = Query(...)):
    """Retrieve configuration for a specific session."""
    try:
        _, config = load_session(session_path)
        return JSONResponse(content=config.to_dict())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    """Convert empty strings to None for optional integer fields."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _attach_session_cookie(response, session_id: str, created: bool) -> None:
    """Persist session id when we had to create a new app."""
    if created:
        response.set_cookie("session_id", session_id, httponly=True, max_age=7200)


@router.post("/preview-dataset")
async def preview_dataset(
    dataset_id: str = Form(...),
    dataset_subset: str = Form(None),
    dataset_split: str = Form("train"),
    dataset_column: str = Form("text"),
    session_id: str = Cookie(None),
):
    """Load dataset and return preview HTML fragment."""
    session_id, app, created = session_manager.get_or_create_app(session_id)

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
        response = HTMLResponse(content=html)
        _attach_session_cookie(response, session_id, created)
        return response

    except Exception as e:
        response = HTMLResponse(content=f"<p><strong>Error:</strong> {str(e)}</p>")
        _attach_session_cookie(response, session_id, created)
        return response


@router.get("/load-model")
async def load_model(
    model_name: str = Query(...),
    context_length: int = Query(4096),
    group_size: int = Query(4),
    engine_name: str = Query("UnSloth (LoRA)"),
    learning_rate: float = Query(2e-5),
    kl_beta: float = Query(0.03),
    temperature: float = Query(1.2),
    max_new_tokens: int = Query(256),
    lora_rank: int = Query(16),
    lora_alpha: int = Query(16),
    max_undos: Optional[str] = Query(None),
    system_prompt: str = Query(""),
    prompt_prefix: str = Query(""),
    prompt_suffix: str = Query(""),
    dataset_id: str = Query("abhayesian/introspection-prompts"),
    precision: str = Query("4-bit"),
    append_think_tag: bool = Query(True),
    use_chat_template: bool = Query(True),
    control_prompt: str = Query(""),
    session_id: str = Cookie(None),
):
    """Load model with SSE status streaming."""
    session_id, app, created = session_manager.get_or_create_app(session_id)

    async def event_stream() -> AsyncIterator[dict]:
        """Stream model loading status."""
        try:
            for status in app.load_model_only(
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
                max_undos=_parse_optional_int(max_undos),
                system_prompt=system_prompt if system_prompt else "",
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                dataset_id=dataset_id,
                precision_choice=precision,
                append_think_tag=append_think_tag,
                use_chat_template=use_chat_template,
                control_prompt=control_prompt,
            ):
                yield {"event": "message", "data": status}

            # Send completion event
            yield {"event": "message", "data": "✅ Model loaded successfully!"}
            yield {"event": "complete", "data": ""}

        except Exception as e:
            yield {"event": "error", "data": f"❌ Error: {str(e)}"}
            yield {"event": "complete", "data": ""}

    response = EventSourceResponse(event_stream())
    _attach_session_cookie(response, session_id, created)
    return response


@router.post("/resume-session")
async def resume_session(
    session_select: str = Form(...),
    session_id: str = Cookie(None),
):
    """Resume an existing session with SSE status streaming."""
    session_id, app, created = session_manager.get_or_create_app(session_id)

    if not session_select:
        response = HTMLResponse(content="<p><strong>Error:</strong> Please select a session</p>")
        _attach_session_cookie(response, session_id, created)
        return response

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

    response = EventSourceResponse(event_stream())
    _attach_session_cookie(response, session_id, created)
    return response


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
    max_undos: Optional[str] = Form(None),
    system_prompt: str = Form(""),
    prompt_prefix: str = Form(""),
    prompt_suffix: str = Form(""),
    dataset_id: str = Form(...),
    dataset_subset: str = Form(None),
    dataset_split: str = Form("train"),
    dataset_column: str = Form("text"),
    precision: str = Form("4-bit"),
    append_think_tag: bool = Form(True),
    use_chat_template: bool = Form(True),
    checkpoint_interval: int = Form(1),
    control_prompt: str = Form(""),
    session_id: str = Cookie(None),
):
    """Start training session with SSE status streaming."""
    session_id, app, created = session_manager.get_or_create_app(session_id)

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
                max_undos=_parse_optional_int(max_undos),
                system_prompt=system_prompt if system_prompt else "",
                prompt_prefix=prompt_prefix,
                prompt_suffix=prompt_suffix,
                dataset_id=dataset_id,
                dataset_subset=dataset_subset if dataset_subset else None,
                dataset_split=dataset_split,
                dataset_column=dataset_column if dataset_column else None,
                precision_choice=precision,
                append_think_tag=append_think_tag,
                use_chat_template=use_chat_template,
                checkpoint_interval=checkpoint_interval,
                control_prompt=control_prompt,
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

    response = EventSourceResponse(event_stream())
    _attach_session_cookie(response, session_id, created)
    return response
