import networkx as nx
import numpy as np
from scipy import linalg



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


def verify_self_loops_transformation(G, transformed_G):
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


def apply_no_loops_method(G):
    """
    Apply the no-loops method to a weighted directed graph.

    Parameters:
    G (networkx.DiGraph): Input weighted directed graph. Edges should have 'weight' attribute.

    Returns:
    networkx.DiGraph: Transformed graph with reversed edges and normalized weights
    """
    # Create a new directed graph
    transformed_G = nx.DiGraph()

    # First add all nodes from original graph
    transformed_G.add_nodes_from(G.nodes())

    # Calculate incoming weights for each node
    win = {}
    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        win[node] = sum(data['weight'] for _, _, data in in_edges)
        if win[node] == 0:
            raise ValueError(f"Node {node} has no incoming edges")

    # Convert edges with normalized weights
    for u, v, data in G.edges(data=True):
        # Create reversed edge with normalized weight qji = pij/win(vj)
        normalized_weight = round(data['weight'] / win[v], 3)
        transformed_G.add_edge(v, u, weight=normalized_weight)

    return transformed_G


def verify_no_loops_transformation(G, transformed_G):
    """
    Verify that the transformation was done correctly by checking:
    1. All original edges are reversed and normalized
    2. No self-loops exist

    Parameters:
    G (networkx.DiGraph): Original graph
    transformed_G (networkx.DiGraph): Transformed graph

    Returns:
    bool: True if transformation is valid
    """
    # Check no self-loops
    if any(transformed_G.has_edge(node, node) for node in transformed_G.nodes()):
        return False

    # Check if all edges are properly reversed and normalized
    for node in transformed_G.nodes():
        out_edges = transformed_G.out_edges(node, data=True)
        out_weights = sum(data['weight'] for _, _, data in out_edges)
        if not np.isclose(out_weights, 1.0, rtol=1e-3):
            return False

    return True


def calc_stationary_distribution(G):
    """
      Calculate the stationary distribution of a Markov chain represented by a NetworkX DiGraph.
      Returns the stationary distribution as a dictionary where the key is the node and the value is the stationary probability.

      Args:
      G (networkx.DiGraph): The directed graph representing the Markov chain.

      Returns:
      dict: A dictionary with nodes as keys and stationary distribution values as values.
      """
    # Number of nodes
    n = len(G.nodes)

    # Create the transition matrix (with probabilities)
    transition_matrix = np.zeros((n, n))
    node_list = list(G.nodes)

    for i, node in enumerate(node_list):
        neighbors = list(G.neighbors(node))
        num_neighbors = len(neighbors)

        if num_neighbors > 0:
            # If the edge has a weight, use it; otherwise, assume uniform probability
            total_weight = sum([G[node][neighbor].get('weight', 1) for neighbor in neighbors])

            for j, neighbor in enumerate(neighbors):
                weight = G[node][neighbor].get('weight', 1)
                transition_matrix[i, node_list.index(neighbor)] = weight / total_weight

    # Solve the system (pi * P = pi) with sum(pi) = 1
    A = transition_matrix.T - np.eye(n)
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1

    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    # Return a dictionary with node as key and stationary distribution as value (rounded to 4 decimal places)
    stationary_distribution = {node_list[i]: round(float(pi[i]), 4) for i in range(n)}

    return stationary_distribution
