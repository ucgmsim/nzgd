"""Edge-list → connected-components helper. Thin wrapper around scipy."""

from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def connected_components_from_edges(
    edges: Iterable[tuple[int, int]],
) -> dict[int, int]:
    """Return a mapping `{node_id: cluster_label}` for each node appearing in any edge.

    Edges are undirected; duplicates and self-loops are ignored. Cluster labels are
    consecutive integers starting at 1. Nodes that do not appear in any edge are not
    in the returned mapping.
    """
    edge_list = [(a, b) for (a, b) in edges if a != b]
    if not edge_list:
        return {}
    nodes_sorted = sorted({n for pair in edge_list for n in pair})
    index_of = {n: i for i, n in enumerate(nodes_sorted)}
    rows = np.fromiter((index_of[a] for a, _ in edge_list), dtype=np.int64)
    cols = np.fromiter((index_of[b] for _, b in edge_list), dtype=np.int64)
    data = np.ones(len(edge_list), dtype=np.int8)
    n = len(nodes_sorted)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(graph, directed=False)
    # Relabel components to 1..n_components in order of first occurrence
    seen: dict[int, int] = {}
    next_id = 1
    out: dict[int, int] = {}
    for node, lbl in zip(nodes_sorted, labels):
        if lbl not in seen:
            seen[int(lbl)] = next_id
            next_id += 1
        out[node] = seen[int(lbl)]
    return out
