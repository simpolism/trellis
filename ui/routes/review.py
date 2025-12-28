"""
Review Routes
=============

Routes for the review/results screen.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Form, Cookie, Query
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from ui.session_manager import session_manager

router = APIRouter(prefix="/review", tags=["review"])


def _attach_session_cookie(response, session_id: str, created: bool) -> None:
    """Persist session id when we had to create a new app."""
    if created:
        response.set_cookie("session_id", session_id, httponly=True, max_age=7200)


@router.post("/save-checkpoint")
async def save_checkpoint(
    checkpoint_name: str = Form(...),
    session_id: str = Cookie(None),
):
    """Save LoRA checkpoint to disk."""
    session_id, app, created = session_manager.get_or_create_app(session_id)

    try:
        result = app.save_checkpoint(checkpoint_name)
        response = HTMLResponse(content=f"""
        <p style="color: var(--trellis-green); font-weight: 600; margin-top: 8px;">
            ✅ {result}
        </p>
        """)
        _attach_session_cookie(response, session_id, created)
        return response
    except Exception as e:
        response = HTMLResponse(content=f"""
        <p style="color: var(--danger); margin-top: 8px;">
            ❌ Error: {str(e)}
        </p>
        """)
        _attach_session_cookie(response, session_id, created)
        return response


@router.get("/merge-lora")
async def merge_lora(
    merge_path: str = Query(...),
    session_id: str = Cookie(None),
):
    """Merge LoRA into base model with SSE progress streaming."""
    session_id, app, created = session_manager.get_or_create_app(session_id)

    async def event_stream() -> AsyncIterator[dict]:
        """Stream merge progress."""
        try:
            yield {"event": "message", "data": "🔄 Merging LoRA into base model..."}
            yield {"event": "message", "data": "This may take several minutes..."}

            # Perform merge (blocking operation)
            result = app.merge_lora(merge_path)

            yield {"event": "message", "data": f"✅ {result}"}
            yield {"event": "complete", "data": ""}

        except Exception as e:
            yield {"event": "error", "data": f"❌ Error: {str(e)}"}
            yield {"event": "complete", "data": ""}

    response = EventSourceResponse(event_stream())
    _attach_session_cookie(response, session_id, created)
    return response
