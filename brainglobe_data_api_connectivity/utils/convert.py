import polars as pl


def convert_matrix_to_edge_table(
    matrix: pl.DataFrame,
    include_zeros: bool = False,
    region_ids: pl.Series | None = None,
) -> pl.DataFrame:
    """Convert adjacency matrix into an edge table.

    Parameters
    matrix: pl.DataFrame
        Adjacency matrix.

    include_zeros : bool (default=False)
        - If True: use only the non-zero entries of the matrix
        - If False: use all entries of the matrix

    region_ids : pl.Series  or None
        Optional mapping from matrix indices to region identifiers.
        If provided, the source and target columns are mapped to these labels.

    Returns
    edge_table : pl.DataFrame
        Table with one row per edge and 3 columns:
            - 1. region identifier (or index) from which the edge originates
            - 2. region identifier (or index) to which the edge points
            - 3. edge weight
    """
    matrix = matrix.clone()
    matrix.columns = [str(i) for i in range(matrix.width)]

    # Convert the adjacency matrix to an edge table.
    edges = (
        matrix.with_row_index("from")
        .unpivot(
            index="from",
            variable_name="to",
            value_name="weight",
        )
        .with_columns(
            pl.col("from", "to").cast(pl.Int64),
        )
        .filter(pl.col("weight").is_not_null())
    )

    if not include_zeros:
        edges = edges.filter(pl.col("weight") != 0)

    if region_ids is not None:
        region_map = dict(enumerate(region_ids.to_list()))
        edges = edges.with_columns(
            pl.col("from", "to").replace_strict(
                region_map,
                return_dtype=region_ids.dtype,
            )
        )

    edges = edges.sort("from", "to")
    return edges
