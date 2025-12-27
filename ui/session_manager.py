"""
Session Manager
===============

Manages HTTP sessions mapped to TrellisApp instances.
Uses in-memory cache for fast access while preserving disk-based persistence.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from threading import Lock

if TYPE_CHECKING:
    from ui.app import TrellisApp


def _get_trellis_app_class():
    """Lazy import of TrellisApp to avoid circular imports."""
    # Import from existing ui/app.py which has the full implementation
    # (even though it has Gradio deps - we'll clean that up later)
    from ui.app import TrellisApp
    return TrellisApp


class SessionManager:
    """
    Manages HTTP sessions with TrellisApp instances.

    - In-memory cache for active sessions
    - Auto-cleanup of inactive sessions after TTL
    - Integrates with existing disk-based persistence
    """

    def __init__(self, base_save_dir: str = "./trellis_sessions", ttl_minutes: int = 120):
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)

        self._sessions: Dict[str, Tuple[object, datetime]] = {}  # session_id -> (TrellisApp, last_access)
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = Lock()  # Thread safety for concurrent requests

    def create_session(self) -> str:
        """
        Create new session with a TrellisApp instance.

        Returns:
            session_id (str): Secure random session identifier
        """
        session_id = secrets.token_urlsafe(32)

        TrellisApp = _get_trellis_app_class()
        app = TrellisApp(base_save_dir=str(self.base_save_dir))

        with self._lock:
            self._sessions[session_id] = (app, datetime.now())

        return session_id

    def get_app(self, session_id: str) -> Optional[object]:
        """
        Retrieve TrellisApp for session, update last access time.

        Args:
            session_id: Session identifier

        Returns:
            TrellisApp instance or None if session doesn't exist
        """
        if not session_id:
            return None

        with self._lock:
            if session_id in self._sessions:
                app, _ = self._sessions[session_id]
                # Refresh timestamp
                self._sessions[session_id] = (app, datetime.now())
                return app

        return None

    def cleanup_expired(self) -> int:
        """
        Remove expired sessions (inactive beyond TTL).

        Returns:
            Number of sessions removed
        """
        now = datetime.now()

        with self._lock:
            expired = [
                sid for sid, (_, last_access) in self._sessions.items()
                if now - last_access > self._ttl
            ]

            for sid in expired:
                del self._sessions[sid]

        return len(expired)

    def session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self._sessions)


# Global session manager instance
session_manager = SessionManager()
