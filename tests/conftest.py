from pathlib import Path

import polars as pl
import pytest

_THIS_DIR = Path(__file__).parent.resolve()


def _to_module_string(path: str) -> str:
    """Convert a file path to a module string."""
    return path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [
    _to_module_string(fixture.relative_to(_THIS_DIR).as_posix())
    for fixture in (Path(f"{_THIS_DIR}/fixtures").rglob("*.py"))
    if "__init__" not in str(fixture)
]


@pytest.fixture
def nodes() -> pl.DataFrame:
    """Simple collection of nodes to use when checking network setup."""
    return pl.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
            "idx": [0, 1, 2, 3],
            "group": ["AB", "AB", "C", "D"],
            "notes": [
                "A is part of group AB",
                "B is part of group AB",
                "C is part of group C",
                "D is part of group D",
            ],
            "custom_index": [1, 2, 3, 4],
        }
    )
