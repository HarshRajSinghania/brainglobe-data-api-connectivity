import polars as pl

from brainglobe_data_api_connectivity.connections import (
    Connections,
)


def test_bidirectional_connections():
    node_info = pl.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
        }
    )
    edge_table = [
        (0, 1, 1.0),  # A -> B
        (1, 0, 1.0),  # B -> A
        (0, 2, 1.0),  # A -> C only
        (3, 0, 1.0),  # D -> A only
    ]

    connections = Connections(
        node_info=node_info,
        edge_table=edge_table,
    )

    result = connections.bidirectional_connections({"name": "A"})

    assert result.get_column("name").to_list() == ["B"]
