"""
Review Routes
=============

Routes for the review/results screen.
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Form, Cookie
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from ui.session_manager import session_manager

router = APIRouter(prefix="/review", tags=["review"])


@router.post("/save-checkpoint")
async def save_checkpoint(
    checkpoint_name: str = Form(...),
    session_id: str = Cookie(None),
):
    """Save LoRA checkpoint to disk."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    try:
        result = app.save_checkpoint(checkpoint_name)
        return HTMLResponse(content=f"""
        <p style="color: var(--trellis-green); font-weight: 600; margin-top: 8px;">
            ✅ {result}
        </p>
        """)
    except Exception as e:
        return HTMLResponse(content=f"""
        <p style="color: var(--danger); margin-top: 8px;">
            ❌ Error: {str(e)}
        </p>
        """)


@router.post("/merge-lora")
async def merge_lora(
    merge_path: str = Form(...),
    session_id: str = Cookie(None),
):
    """Merge LoRA into base model with SSE progress streaming."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

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

    return EventSourceResponse(event_stream())
