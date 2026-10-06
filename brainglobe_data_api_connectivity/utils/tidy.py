"""Tidy Excel input data."""

import re
from collections.abc import Iterable


def rename_columns(columns: Iterable[str]) -> list[str]:
    """Return standardised column names as a list.

    Names contain lowercase letters, digits and underscores. Leading digits
    move to the end, and leading or trailing underscores are removed.
    """
    cleaned = []
    for column in columns:
        name = re.sub(r"[^a-z0-9]+", "_", column.lower())
        name = re.sub(r"^(\d+)([a-z0-9_]+)$", r"\2_\1", name)
        cleaned.append(re.sub(r"_+", "_", name).strip("_"))
    return cleaned
