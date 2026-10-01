import polars as pl

from brainglobe_data_api_connectivity.connections import (
    Connections,
)


def test_no_bidirectional_connections():
    node_info = pl.DataFrame(
        {
            "name": ["A", "B"],
        }
    )

    connections = Connections(
        node_info=node_info,
        edge_table=[
            (0, 1, 2.0),
        ],
    )

    has_connections, connection_info = connections.bidirectional_connections(
        {"name": "A"}
    )

    assert not has_connections
    assert connection_info.is_empty()


def test_bidirectional_connections():
    node_info = pl.DataFrame(
        {
            "name": ["A", "B", "C", "D"],
        }
    )

    connections = Connections(
        node_info=node_info,
        edge_table=[
            (0, 1, 2.0),  # A -> B
            (1, 0, 3.0),  # B -> A
            (0, 2, 4.0),  # A -> C
            (2, 0, 6.0),  # C -> A
            (3, 0, 5.0),  # D -> A, directional only
        ],
    )

    has_connections, connection_info = connections.bidirectional_connections(
        {"name": "A"}
    )

    assert has_connections

    expected = pl.DataFrame(
        {
            "from": ["A", "B", "A", "C"],
            "to": ["B", "A", "C", "A"],
            "strength": [2.0, 3.0, 4.0, 6.0],
        }
    )

    assert connection_info.equals(expected)
