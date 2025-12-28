#!/usr/bin/env python3
"""
TRELLIS - Interactive Preference Steering for Language Models
==============================================================

FastAPI + HTMX implementation

Usage:
    python trellis.py                # Launch server on port 7860
    python trellis.py --port 7861    # Custom port
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent))

_WATCHFILES_IGNORE_PATTERN = "unsloth_compiled_cache/*"
_WATCHFILES_IGNORE_DIR_PATTERN = "unsloth_compiled_cache/**"
_WATCHFILES_IGNORE_ROOT = "unsloth_compiled_cache"
_WATCHFILES_IGNORE_ANY_PATTERN = "*/unsloth_compiled_cache/*"
_WATCHFILES_IGNORE_ANY_DIR_PATTERN = "*/unsloth_compiled_cache/**"
_WATCHFILES_IGNORE_ANY_ROOT = "*/unsloth_compiled_cache"
_RELOAD_EXCLUDES = [
    _WATCHFILES_IGNORE_ROOT,
    _WATCHFILES_IGNORE_PATTERN,
    _WATCHFILES_IGNORE_DIR_PATTERN,
    _WATCHFILES_IGNORE_ANY_ROOT,
    _WATCHFILES_IGNORE_ANY_PATTERN,
    _WATCHFILES_IGNORE_ANY_DIR_PATTERN,
]
_existing_ignore = os.environ.get("WATCHFILES_IGNORE")
if _existing_ignore:
    extra = ",".join(_RELOAD_EXCLUDES)
    if extra not in _existing_ignore:
        os.environ["WATCHFILES_IGNORE"] = f"{_existing_ignore},{extra}"
else:
    os.environ["WATCHFILES_IGNORE"] = ",".join(_RELOAD_EXCLUDES)

import uvicorn
from fastapi import FastAPI, Request, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import session manager and routes
from ui.session_manager import session_manager
from ui.routes import setup_router, training_router, review_router

# Create FastAPI app
app = FastAPI(title="Trellis", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="ui/templates")

# Include route modules
app.include_router(setup_router)
app.include_router(training_router)
app.include_router(review_router)


# ========== Main Screen Routes ==========

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect root to setup."""
    return RedirectResponse(url="/setup")


@app.get("/setup", response_class=HTMLResponse)
async def setup_screen(request: Request, session_id: str = Cookie(None)):
    """Setup screen - create session if needed."""
    session_id, app_instance, new_session = session_manager.get_or_create_app(session_id)

    # Get existing sessions for resume dropdown
    sessions = []
    if app_instance:
        try:
            sessions = app_instance.discover_sessions()
        except:
            sessions = []

    response = templates.TemplateResponse("setup.html", {
        "request": request,
        "active_tab": "setup",
        "sessions": sessions,
        "default_dataset": "abhayesian/introspection-prompts",
        "session_id": session_id,
    })
    if new_session:
        response.set_cookie("session_id", session_id, httponly=True, max_age=7200)
    return response


@app.get("/training", response_class=HTMLResponse)
async def training_screen(request: Request, session_id: str = Cookie(None)):
    """Training screen."""
    app_instance = session_manager.get_app(session_id)

    if not app_instance or not app_instance.model_loaded:
        return RedirectResponse(url="/setup")

    stats = app_instance.get_stats() if hasattr(app_instance, 'get_stats') else ("**Step:** 0", "**Drift:** 0.000", "**Dataset:** Not loaded")
    group_size = 4
    if hasattr(app_instance, "config") and app_instance.config:
        group_size = app_instance.config.group_size

    return templates.TemplateResponse("training.html", {
        "request": request,
        "active_tab": "training",
        "step": stats[0],
        "drift": stats[1],
        "dataset": stats[2],
        "group_size": group_size,
    })


@app.get("/review", response_class=HTMLResponse)
async def review_screen(request: Request, session_id: str = Cookie(None)):
    """Review screen."""
    app_instance = session_manager.get_app(session_id)

    if not app_instance:
        return RedirectResponse(url="/setup")

    journal = ""
    config = {}
    steps = "**Total Steps:** 0"
    drift = "**Final Drift:** 0.000"

    if hasattr(app_instance, 'get_journal_content'):
        journal = app_instance.get_journal_content()
    if hasattr(app_instance, 'get_config_display'):
        config = app_instance.get_config_display()
    if hasattr(app_instance, 'get_final_stats'):
        stats = app_instance.get_final_stats()
        steps = stats[0]
        drift = stats[1]

    return templates.TemplateResponse("review.html", {
        "request": request,
        "active_tab": "review",
        "journal": journal,
        "config": config,
        "steps": steps,
        "drift": drift,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Trellis: Interactive Preference Steering (FastAPI)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trellis_sessions",
        help="Directory for session data (default: ./trellis_sessions)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    args = parser.parse_args()

    # Update session manager base directory
    session_manager.base_save_dir = Path(args.save_dir)
    session_manager.base_save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TRELLIS - Interactive Preference Steering (HTMX)")
    print("=" * 60)
    print()
    print(f"Sessions directory: {args.save_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print()

    # Run with uvicorn
    uvicorn.run(
        "trellis:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_excludes=_RELOAD_EXCLUDES if args.reload else None,
    )


if __name__ == "__main__":
    main()
