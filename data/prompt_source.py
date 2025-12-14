"""
Prompt Source
=============

Loads prompts from HuggingFace datasets with auto-detection of text columns.
"""

from __future__ import annotations

import random
from typing import Optional


class PromptSource:
    """
    Loads prompts from HuggingFace datasets.

    Attempts to auto-detect the text column, or uses explicit column name.
    Supports shuffling and subset selection.
    """

    # Common column names for prompts in various datasets
    CANDIDATE_COLUMNS = [
        "question", "prompt", "instruction", "input", "text",
        "query", "content", "message", "human", "user",
    ]

    def __init__(self):
        self.dataset = None
        self.dataset_id: Optional[str] = None
        self.subset: Optional[str] = None
        self.split: str = "train"
        self.prompts: list[str] = []
        self.index: int = 0
        self.text_column: Optional[str] = None

    def load(
        self,
        dataset_id: str,
        subset: Optional[str] = None,
        split: str = "train",
        text_column: Optional[str] = None,
        shuffle: bool = True,
        max_prompts: Optional[int] = None,
    ) -> str:
        """
        Load a dataset from HuggingFace.

        Args:
            dataset_id: HF dataset ID (e.g., "Anthropic/model-written-evals")
            subset: Dataset subset/config name (e.g., "persona")
            split: Which split to use (default: "train")
            text_column: Column containing prompts (auto-detected if None)
            shuffle: Whether to shuffle prompts
            max_prompts: Limit number of prompts (None for all)

        Returns:
            Status message
        """
        try:
            from datasets import load_dataset
        except ImportError:
            return "Install datasets: pip install datasets"

        try:
            # Load dataset
            if subset:
                ds = load_dataset(dataset_id, subset, split=split)
            else:
                ds = load_dataset(dataset_id, split=split)

            self.dataset = ds
            self.dataset_id = dataset_id
            self.subset = subset
            self.split = split

            # Auto-detect or validate text column
            columns = ds.column_names

            if text_column:
                if text_column not in columns:
                    return f"Column '{text_column}' not found. Available: {columns}"
                self.text_column = text_column
            else:
                # Try to auto-detect
                self.text_column = self._detect_text_column(columns)
                if not self.text_column:
                    return f"Could not detect text column. Available: {columns}. Specify manually."

            # Extract prompts
            self.prompts = [str(row[self.text_column]) for row in ds if row[self.text_column]]

            # Limit if requested
            if max_prompts and len(self.prompts) > max_prompts:
                self.prompts = self.prompts[:max_prompts]

            # Shuffle if requested
            if shuffle:
                random.shuffle(self.prompts)

            self.index = 0

            subset_str = f"/{subset}" if subset else ""
            return f"Loaded {len(self.prompts)} prompts from {dataset_id}{subset_str} (column: {self.text_column})"

        except Exception as e:
            return f"Failed to load dataset: {e}"

    def preview(self, n: int = 3) -> list[str]:
        """Get a preview of the first n prompts without advancing."""
        return self.prompts[:n] if self.prompts else []

    def _detect_text_column(self, columns: list[str]) -> Optional[str]:
        """Try to find the text column automatically."""
        # Check exact matches first (case-insensitive)
        columns_lower = {c.lower(): c for c in columns}
        for candidate in self.CANDIDATE_COLUMNS:
            if candidate in columns_lower:
                return columns_lower[candidate]

        # Check partial matches
        for candidate in self.CANDIDATE_COLUMNS:
            for col in columns:
                if candidate in col.lower():
                    return col

        # Fall back to first string-looking column
        return columns[0] if columns else None

    def next(self) -> Optional[str]:
        """Get the next prompt, cycling if exhausted."""
        if not self.prompts:
            return None

        prompt = self.prompts[self.index]
        self.index = (self.index + 1) % len(self.prompts)
        return prompt

    def peek(self) -> Optional[str]:
        """See current prompt without advancing."""
        if not self.prompts:
            return None
        return self.prompts[self.index]

    def skip(self, n: int = 1):
        """Skip forward n prompts."""
        if self.prompts:
            self.index = (self.index + n) % len(self.prompts)

    def reset(self, shuffle: bool = True):
        """Reset to beginning, optionally reshuffling."""
        if shuffle and self.prompts:
            random.shuffle(self.prompts)
        self.index = 0

    def remaining(self) -> int:
        """How many prompts until we cycle."""
        return len(self.prompts) - self.index if self.prompts else 0

    def total(self) -> int:
        """Total number of prompts."""
        return len(self.prompts)

    def is_loaded(self) -> bool:
        """Whether a dataset is loaded."""
        return len(self.prompts) > 0

    def status(self) -> str:
        """Current status string."""
        if not self.prompts:
            return "No dataset loaded"
        return f"{self.dataset_id}: {self.index + 1}/{len(self.prompts)}"

    def get_state(self) -> dict:
        """Get serializable state for session persistence."""
        return {
            "dataset_id": self.dataset_id,
            "subset": self.subset,
            "split": self.split,
            "text_column": self.text_column,
            "index": self.index,
        }

    def restore_state(self, state: dict) -> str:
        """Restore from saved state. Returns status message."""
        if not state.get("dataset_id"):
            return "No dataset in saved state"

        result = self.load(
            dataset_id=state["dataset_id"],
            subset=state.get("subset"),
            split=state.get("split", "train"),
            text_column=state.get("text_column"),
            shuffle=False,  # Don't shuffle on restore
        )

        if "Failed" not in result and "not found" not in result:
            self.index = state.get("index", 0)

        return result
