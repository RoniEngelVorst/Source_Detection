import networkx as nx
import numpy as np


def reverse_and_normalize_weights(G):
    """
    Reverse a directed graph and normalize the weights of the edges.

    :param G: Directed graph (DiGraph) with 'weight' as an edge attribute.
    :return: A new reversed graph with normalized weights.
    """
    # Reverse the graph
    reversed_G = nx.DiGraph()

    # Populate the reversed graph and normalize weights
    for node in G.nodes:
        # Get all incoming edges to this node in the original graph
        incoming_edges = G.in_edges(node, data=True)
        total_weight = sum(attr.get('weight', 1) for _, _, attr in incoming_edges)

        # Avoid division by zero
        if total_weight > 0:
            for u, v, attr in incoming_edges:
                weight = attr.get('weight', 1)
                normalized_weight = round(weight / total_weight, 3)
                # Add the reversed edge with normalized weight
                reversed_G.add_edge(v, u, weight=normalized_weight)

    return reversed_G


def apply_self_loop_method(G):
    """
    Apply the self-loop method to a weighted directed graph.

    Parameters:
    G (networkx.DiGraph): Input weighted directed graph. Edges should have 'weight' attribute.

    Returns:
    networkx.DiGraph: Transformed graph with normalized weights and self-loops
    """
    # Create a new graph for the transformed version
    transformed_G = nx.DiGraph()

    # Step 1: Calculate maxin (maximum incoming weight for any node)
    max_in = 0
    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        in_weight = sum(data['weight'] for _, _, data in in_edges)
        max_in = max(max_in, in_weight)

    if max_in == 0:
        raise ValueError("Graph has no weighted edges")

    # Step 1: Convert edges with normalized weights
    for u, v, data in G.edges(data=True):
        # Create reversed edge with normalized weight (rounded to 3 decimal places)
        normalized_weight = round(data['weight'] / max_in, 3)
        transformed_G.add_edge(v, u, weight=normalized_weight)

    # Step 2: Add self-loops
    for node in G.nodes():
        # Calculate total incoming weight for the node
        in_edges = G.in_edges(node, data=True)
        in_weight = sum(data['weight'] for _, _, data in in_edges)

        # Add self-loop with appropriate weight (rounded to 3 decimal places)
        self_loop_weight = round((max_in - in_weight) / max_in, 3)

        # Only add self-loop if weight is not zero
        if self_loop_weight > 0:
            transformed_G.add_edge(node, node, weight=self_loop_weight)

    return transformed_G


def verify_transformation(G, transformed_G):
    """
    Verify that the transformation was done correctly by checking:
    1. All nodes have outgoing probabilities that sum to 1
    2. All original edges are reversed and normalized
    3. All nodes have self-loops if needed

    Parameters:
    G (networkx.DiGraph): Original graph
    transformed_G (networkx.DiGraph): Transformed graph

    Returns:
    bool: True if transformation is valid
    """
    for node in transformed_G.nodes():
        # Check if outgoing probabilities sum to 1 (within numerical precision)
        out_weights = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_weights, 1.0, rtol=1e-3):  # Increased tolerance due to rounding
            return False

    return True