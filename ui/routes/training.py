"""
Training Routes
===============

Routes for the training screen with SSE streaming.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Form, Cookie
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from ui.session_manager import session_manager

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/generate-options")
async def generate_options(session_id: str = Cookie(None)):
    """Generate options with SSE streaming."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    async def event_stream() -> AsyncIterator[dict]:
        """Stream prompt and options as they generate."""
        try:
            # Get next prompt
            prompt = app.prompt_source.next()
            if not prompt:
                yield {"event": "error", "data": "No prompts available"}
                yield {"event": "complete", "data": ""}
                return

            app.current_prompt = prompt

            # Send prompt immediately
            yield {"event": "prompt", "data": prompt}

            # Stream options as they generate
            if hasattr(app.engine, "generate_options_streaming"):
                for options in app.engine.generate_options_streaming(prompt):
                    app.current_options = options
                    yield {"event": "options", "data": json.dumps(options)}
            else:
                # Fallback: blocking generation
                options = app.engine.generate_options(prompt)
                app.current_options = options
                yield {"event": "options", "data": json.dumps(options)}

            # Log to journal
            if app.journal:
                app.journal.log_generation(prompt, len(app.current_options))

            # Get updated stats
            stats = app.get_stats()
            yield {"event": "stats", "data": json.dumps({
                "step": stats[0],
                "drift": stats[1],
                "dataset": stats[2],
            })}

            yield {"event": "complete", "data": ""}

        except Exception as e:
            yield {"event": "error", "data": f"Error: {str(e)}"}
            yield {"event": "complete", "data": ""}

    return EventSourceResponse(event_stream())


@router.post("/select-option/{choice_idx}")
async def select_option(choice_idx: int, session_id: str = Cookie(None)):
    """Select an option, train, and return HTML fragment."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    try:
        # Train on selection
        status, metrics = app.select_option(choice_idx)

        # Return success message
        return HTMLResponse(content=f"""
        <div style="color: var(--trellis-green); font-weight: 600;">
            ✅ Trained on option {chr(65 + choice_idx)}
        </div>
        """)

    except Exception as e:
        return HTMLResponse(content=f"<p>Error: {str(e)}</p>")


@router.post("/skip")
async def skip_prompt(session_id: str = Cookie(None)):
    """Skip current prompt without training."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    try:
        app.skip_prompt()
        return HTMLResponse(content="""
        <div style="color: var(--gray-dark); font-weight: 600;">
            ⏭️ Skipped
        </div>
        """)
    except Exception as e:
        return HTMLResponse(content=f"<p>Error: {str(e)}</p>")


@router.post("/undo")
async def undo(session_id: str = Cookie(None)):
    """Undo to previous checkpoint and return HTML fragment."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    try:
        checkpoint, status = app.undo()

        if not checkpoint:
            return HTMLResponse(content=f"<p>{status}</p>")

        # Get stats after undo
        stats = app.get_stats()

        # Return HTML fragment with restored state
        html = f"""
        <div>
            <div class="stats-header">
                <div>{stats[0]}</div>
                <div>{stats[1]}</div>
                <div>{stats[2]}</div>
            </div>
            <hr>
            <div class="prompt-card">
                <strong>Prompt:</strong><br><br>
                {checkpoint.prompt if checkpoint.prompt else ''}
            </div>
            <hr>
            <div id="options-container">
                <h3>Select your preference:</h3>
        """

        # Add option cards
        if checkpoint.options:
            for i, option in enumerate(checkpoint.options[:4]):
                label = chr(65 + i)
                html += f"""
                <div class="option-group">
                    <div class="option-header">{label}</div>
                    <div class="option-text">{option}</div>
                    <button class="option-btn primary"
                            hx-post="/training/select-option/{i}"
                            hx-target="#action-status"
                            hx-swap="innerHTML">
                        Select {label}
                    </button>
                </div>
                """

        html += """
            </div>
            <div id="action-status" style="margin-top: 16px; color: var(--trellis-green); font-weight: 600;">
                ↩️ """ + status + """
            </div>
        </div>
        """

        return HTMLResponse(content=html)

    except Exception as e:
        return HTMLResponse(content=f"<p>Error: {str(e)}</p>")


@router.post("/edit-prompt")
async def edit_prompt(
    new_prompt: str = Form(...),
    session_id: str = Cookie(None),
):
    """Regenerate with edited prompt."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    try:
        prompt, options = app.apply_edited_prompt(new_prompt)

        # Return HTML fragment with new options
        html = "<h3>Select your preference:</h3>"
        for i, option in enumerate(options[:4]):
            label = chr(65 + i)
            html += f"""
            <div class="option-group">
                <div class="option-header">{label}</div>
                <div class="option-text">{option}</div>
                <button class="option-btn primary"
                        hx-post="/training/select-option/{i}"
                        hx-target="#action-status"
                        hx-swap="innerHTML">
                    Select {label}
                </button>
            </div>
            """

        return HTMLResponse(content=html)

    except Exception as e:
        return HTMLResponse(content=f"<p>Error: {str(e)}</p>")


@router.post("/save-session")
async def save_session(
    session_name: str = Form(...),
    session_id: str = Cookie(None),
):
    """Save session with custom name."""
    app = session_manager.get_app(session_id)
    if not app:
        return HTMLResponse(content="<p>Error: No session found</p>")

    try:
        result = app.save_session_with_name(session_name)
        return HTMLResponse(content=f"<p style='color: var(--trellis-green);'>✅ {result}</p>")
    except Exception as e:
        return HTMLResponse(content=f"<p>Error: {str(e)}</p>")
