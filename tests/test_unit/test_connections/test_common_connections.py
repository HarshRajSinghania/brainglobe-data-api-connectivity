import polars as pl
import pytest

from brainglobe_data_api_connectivity.connections import Connections
from brainglobe_data_api_connectivity.connections.query_opts import (
    ConnectionsLookup,
    NodeIs,
)


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


@pytest.fixture
def edge_list() -> list[tuple[int, int, float]]:
    """Edge list fixture compatible with Connections().

    Representation of the simple network with nodes A (0), B (1), C (2), and
    D (3):

    (A) ── 0.1 ──▶ (B) ── 10.0 ──▶ (C) ── 10.0 ──▶ (D)
     │                               ▲
     │                               │
     └───────────── 1.0 ─────────────┘

    """
    return [
        (0, 1, 0.1),
        (1, 2, 1.0),
        (0, 2, 10),
        (2, 3, 10),
    ]


@pytest.fixture
def edge_info() -> pl.DataFrame:
    "A simple 'edge information' frame for testing setup."
    return pl.DataFrame(
        {
            "from_id": ["A", "B", "A", "A", "C"],
            "to_id": ["B", "C", "C", "C", "D"],
            "from": [0, 1, 0, 0, 2],
            "to": [1, 2, 2, 2, 3],
            "used": ["yes", "yes", "yes", "no", "yes"],
            "paper": [
                "author et al., 1998",
                "author et al., 2025",
                "author et al., 2010",
                "author et al., 2020",
                "author et al., 2020",
            ],
            "strength": [
                "weak (0.1)",
                "medium (1.0)",
                "strong (10.0)",
                "medium (1.0)",
                "strong (10.0)",
            ],
        }
    )


@pytest.fixture
def mini_G(nodes, edge_list, edge_info) -> Connections:
    """Small Connections instance for testing"""
    return Connections(nodes, edge_list, edge_info)


@pytest.mark.parametrize(
    ("node_indices", "node_as", "expected"),
    [
        pytest.param(
            [0, 1],
            NodeIs.ANY,
            [2],
            id="common connections in either direction",
        ),
        pytest.param(
            [0, 1], NodeIs.INPUT, [2], id="common output of nodes 0 and 1"
        ),
        pytest.param(
            [1, 2], NodeIs.OUTPUT, [0], id="common input to nodes 1 and 2"
        ),
    ],
)
@pytest.mark.parametrize(
    "connections_lookup",
    [
        pytest.param(ConnectionsLookup.ALL, id="all edge information"),
        pytest.param(ConnectionsLookup.REPORTED, id="in network"),
    ],
)
def test_common_connections(
    mini_G,
    node_indices,
    node_as,
    expected,
    connections_lookup,
) -> None:
    """Return nodes directly connected to all given nodes."""
    common = mini_G.common_connections(
        node_indices,
        node_as=node_as,
        connections_lookup=connections_lookup,
    )
    assert common == expected


@pytest.mark.parametrize(
    ("edge_table", "node_as"),
    [
        pytest.param(
            [(1, 3, 1.0), (2, 3, 1.0)],
            NodeIs.INPUT,
            id="nodes 1 and 2 are inputs to 3",
        ),
        pytest.param(
            [(3, 1, 1.0), (3, 2, 1.0)],
            NodeIs.OUTPUT,
            id="nodes 1 and 2 are outputs of 3",
        ),
    ],
)
def test_common_connections_direction_examples(edge_table, node_as) -> None:
    """Nodes 1 and 2 share node 3 in the specified direction only."""
    graph = Connections(
        node_info=pl.DataFrame({"idx": [0, 1, 2, 3]}),
        edge_table=edge_table,
    )

    assert graph.common_connections([1, 2], node_as=node_as) == [3]
