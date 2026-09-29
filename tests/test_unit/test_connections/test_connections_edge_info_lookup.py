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
            "from": ["A", "B", "A", "A", "C"],
            "to": ["B", "C", "C", "C", "D"],
            "source_node_idx": [0, 1, 0, 0, 2],  # not always present
            "target_node_idx": [1, 2, 2, 2, 3],  # not always present
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
    """"""
    return Connections(nodes, edge_list, edge_info)


@pytest.mark.parametrize(
    ["node0", "node1", "node0_as", "expected_bool", "expected_shape"],
    [
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.ANY,
            True,
            (1, 9),
            id="A to B",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "C"},
            NodeIs.ANY,
            True,
            (2, 9),
            id="A to C",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.ANY,
            True,
            (1, 9),
            id="B to A",
        ),
        pytest.param(
            {"name": "B"},
            {"name": "A"},
            NodeIs.INPUT,
            False,
            (0, 9),
            id="B to A input only (no direct connection)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.INPUT,
            True,
            (1, 9),
            id="A to B input only",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "B"},
            NodeIs.OUTPUT,
            False,
            (0, 9),
            id="A to B output only (no direct connection)",
        ),
        pytest.param(
            {"name": "A"},
            {"name": "D"},
            NodeIs.ANY,
            False,
            (0, 9),
            id="A to D (no direct connection)",
        ),
    ],
)
def test_has_direct_connection_between(
    mini_G, node0, node1, node0_as, expected_bool, expected_shape
):
    has_connection, connections = mini_G.direct_connection_between(
        node0,
        node1,
        node0_as=node0_as,
        connections_lookup=ConnectionsLookup.ALL,
    )

    assert has_connection == expected_bool
    assert connections.shape == expected_shape
