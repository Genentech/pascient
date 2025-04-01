import networkx as nx
from typing import Dict
import numpy as np


def build_ancestry_matrix(
    onto: nx.Graph, node_id_to_int: Dict[str, int]
) -> np.array:
    """
    Build a matrix where entry `[i, j]` is 1 if `i` is the ancestor of `j` or vice-versa.
    i is considered to be ancestor of i (so the main diagonal is all 1s)
    """
    
    # make sure no empty rows of the ancestry matrix
    assert set(node_id_to_int.values()) == set(range(len(node_id_to_int)))
    n = len(node_id_to_int)
    # build dictionary of which cells have which ancestors
    all_desc = {}
    labeled_node_ints = []
    for node in onto.nodes:
        node_int = node_id_to_int.get(node, None)
        if node_int is None:  # not all nodes are labeled
            continue
        labeled_node_ints.append(node_int)
        descendent_names = nx.descendants(onto, node)
        descendent_ints = {
            node_id_to_int[name] for name in descendent_names
            if name in node_id_to_int
        }
        all_desc[node_int] = descendent_ints
    
    # we count nodes as their own ancestors
    ancestry = np.eye(n, dtype=np.int8)

    for i, node_1 in enumerate(labeled_node_ints):
        for j, node_2 in enumerate(labeled_node_ints):
            if i == j:
                continue
            ancestry[i, j] = (
                # node_1 descendent of node_2?
                node_2 in all_desc[node_1]
                # node_2 descendent of node_1?
                or node_1 in all_desc[node_2]
            )
    #ancestry[i,j] = 1 iff i is ancestor of j or vice-versa
    return ancestry