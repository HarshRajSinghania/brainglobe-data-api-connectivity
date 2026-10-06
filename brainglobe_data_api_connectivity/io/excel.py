"""Excel-related helper functions."""

import re
from pathlib import Path

import polars as pl
from fastexcel import read_excel


def get_df_from_excel(
    file: Path,
    sheet_name: str | None = None,
    data_range: tuple[str, str] | None = None,
    header: int | list[int] | None = None,
) -> pl.DataFrame:
    """Return DataFrame sliced to given row/column ranges."""
    columns, start_row, n_rows = None, 0, None
    if data_range is not None:
        (first_col, last_col), (first_row, last_row) = get_cell_range(
            data_range
        )
        columns = list(range(first_col, last_col + 1))
        start_row, n_rows = first_row - 1, last_row - first_row + 1

    reader = read_excel(file)
    sheet = sheet_name or 0
    names = None
    if isinstance(header, list):
        consumed_rows = max(header) + 1
        headings = reader.load_sheet(
            sheet,
            header_row=None,
            skip_rows=start_row,
            n_rows=consumed_rows,
            use_columns=columns,
        ).to_polars()
        rows = []
        for i in header:
            row = headings.row(i)
            if i != header[-1]:
                row = pl.Series(row).fill_null(strategy="forward").to_list()
            rows.append(row)
        names = [
            "_".join(str(v) for v in values if v is not None)
            for values in zip(*rows)
        ]
        header = None
    else:
        consumed_rows = header + 1 if header is not None else 0

    table = reader.load_sheet(
        sheet,
        header_row=start_row + header if header is not None else None,
        skip_rows=0 if header is not None else start_row + consumed_rows,
        n_rows=n_rows - consumed_rows if n_rows is not None else None,
        use_columns=columns,
        schema_sample_rows=None,
    ).to_polars()
    if names is not None or header is None:
        table.columns = names or [str(i) for i in range(table.width)]
    return table


def validate_cell_reference(ref: str) -> None:
    """Validate that a cell reference is letters A–Z followed by digits 0–9."""

    pattern = r"^[A-Za-z]+[0-9]+$"

    if not re.match(pattern, ref):
        raise ValueError(f"Invalid cell reference: {ref}")


def cell_reference_to_indices(ref: str) -> tuple[int, int]:
    """Split a cell reference into zero-based column index and row number."""
    validate_cell_reference(ref)
    col_ref = "".join(filter(str.isalpha, ref))
    row_ref = "".join(filter(str.isdigit, ref))
    return column_reference_to_index(col_ref), int(row_ref)


def column_reference_to_index(label: str) -> int:
    """converts Excel column labels into numeric indices."""
    label = label.upper()

    index = 0
    for char in label:
        index = index * 26 + (ord(char) - ord("A") + 1)

    return index - 1


def normalise_index_range(
    start: tuple[int, int],
    end: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Normalise two (col, row) index pairs to top-left → bottom-right.

    Examples:
        ((3, 1), (1, 3)) becomes ((1, 1), (3, 3))
        ((1, 3), (3, 1)) becomes ((1, 1), (3, 3))
        ((3, 3), (1, 1)) becomes ((1, 1), (3, 3))
        ((1, 1), (3, 3)) stays ((1, 1), (3, 3))
    """
    (col_a, row_a), (col_b, row_b) = start, end

    min_col = min(col_a, col_b)
    max_col = max(col_a, col_b)
    min_row = min(row_a, row_b)
    max_row = max(row_a, row_b)

    return (min_col, min_row), (max_col, max_row)


def validate_cell_range(cell_range: tuple[str, str]) -> None:
    """Validate that cell_range contains two cell references."""
    if len(cell_range) != 2:
        raise ValueError("Cell range must contain two cell references.")


def get_cell_range(
    cell_range: tuple[str, str],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert two cell references into column and row index ranges."""
    validate_cell_range(cell_range)

    start = cell_reference_to_indices(cell_range[0])
    end = cell_reference_to_indices(cell_range[1])

    (start_col, start_row), (end_col, end_row) = normalise_index_range(
        start, end
    )

    return (start_col, end_col), (start_row, end_row)
