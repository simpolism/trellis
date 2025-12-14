"""
Trellis State Module
====================

Manages training state with linear undo/redo and session persistence.
"""

from .checkpoint import Checkpoint
from .undo_stack import LinearUndoStack
from .session import SessionState

__all__ = ["Checkpoint", "LinearUndoStack", "SessionState"]
