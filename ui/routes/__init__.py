"""
Trellis FastAPI Routes
======================

Route modules for the Trellis application.
"""

from .setup import router as setup_router
from .training import router as training_router
from .review import router as review_router

__all__ = ["setup_router", "training_router", "review_router"]
