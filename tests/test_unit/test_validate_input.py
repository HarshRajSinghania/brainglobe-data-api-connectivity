import polars as pl
import pytest

from brainglobe_data_api_connectivity.io.validate_input import (
    validate_adjacency_matrix,
)


@pytest.mark.parametrize(
    ["matrix", "error"],
    [
        pytest.param(
            pl.DataFrame(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                orient="row",
            ),
            None,
            id="valid",
        ),
        pytest.param(
            pl.DataFrame(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                orient="row",
            ),
            ValueError("matrix must be square, but got 5 rows and 4 columns."),
            id="Adjacency matrix should be square",
        ),
    ],
)
def test_validate_adjacency_matrix(matrix, error, raises_error):
    if error is None:
        validate_adjacency_matrix(matrix)
    else:
        with raises_error(error):
            validate_adjacency_matrix(matrix)
