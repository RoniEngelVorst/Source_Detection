import heapq

import networkx as nx
import numpy as np
from fontTools.merge.util import equal
from scipy import linalg
from scipy.sparse.linalg import eigs
import random
import time


def reverse_and_normalize_weights(G):
    """
    Reverse a directed graph and normalize the weights of the edges.

    :param G: Directed graph (DiGraph) with 'weight' as an edge attribute.
    :return: A new reversed graph with normalized weights.
    """
    # Reverse the graph
    reversed_G = G.reverse(copy=True)

    # Populate the reversed graph and normalize weights
    for node in G.nodes:
        # Get all incoming edges to this node in the original graph
        incoming_edges = G.in_edges(node, data=True)
        total_weight = sum(attr.get('weight', 1) for _, _, attr in incoming_edges)

        # Avoid division by zero
        if total_weight > 0:
            for u, v, attr in incoming_edges:
                weight = attr.get('weight', 1)
                normalized_weight = weight / total_weight
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

    transformed_G = G.reverse(copy=True)
    max_in = max(
        sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
        for node in G.nodes()
    )

    if max_in == 0:
        # raise ValueError("Graph has no weighted edges")
        return transformed_G

    for node in G.nodes():
        in_weight = sum(data['weight'] for _, _, data in G.in_edges(node, data=True))
        self_loop_weight = (max_in - in_weight) / max_in

        if self_loop_weight > 0:
            transformed_G.add_edge(node, node, weight=self_loop_weight)

    for u, v, data in G.edges(data=True):
        normalized_weight = data['weight'] / max_in
        transformed_G.add_edge(v, u, weight=normalized_weight)

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
        out_weights = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_weights, 1.0, atol=1e-3):
            print(f"Verification failed: Node {node} has outgoing sum {out_weights}, expected 1.")
            print(f"Outgoing edges for {node}: {list(transformed_G.out_edges(node, data=True))}")
            return False

    return True

def apply_no_loops_new(G):
    # naive_G = reverse_and_normalize_weights(G)
    #
    # for u, v, data in naive_G.edges(data=True):
    #     in_edges = G.in_edges(v, data=True)
    #     win = sum(data['weight'] for _, _, data in in_edges)  # Compute win(vj)
    #
    #     if win > 0:
    #         naive_G.edges[u, v]['weight'] /= win
    #     else:
    #         print(f"Warning: Node {v} has zero in-degree (win = 0). Assigning uniform distribution.")
    #         num_outgoing = len(list(naive_G.out_edges(v)))
    #         if num_outgoing > 0:
    #             naive_G.edges[u, v]['weight'] = 1.0 / num_outgoing  # Evenly distribute
    #
    #     # **Final Check: Normalize Outgoing Weights**
    # for node in naive_G.nodes():
    #     out_weight_sum = sum(data['weight'] for _, _, data in naive_G.out_edges(node, data=True))
    #
    #     if out_weight_sum > 0:
    #         for _, v, data in naive_G.out_edges(node, data=True):
    #             data['weight'] /= out_weight_sum  # Normalize outgoing edges
    #
    # return naive_G
    transformed_G = G.reverse(copy=True)  # Reverse edges

    for u, v, data in transformed_G.edges(data=True):
        in_edges = G.in_edges(v, data=True)
        win = sum(data['weight'] for _, _, data in in_edges)  # Compute win(vj)

        if win > 0:
            transformed_G.edges[u, v]['weight'] /= win
        else:
            print(f"Warning: Node {v} has zero in-degree (win = 0). Assigning uniform distribution.")
            num_outgoing = len(list(transformed_G.out_edges(v)))
            if num_outgoing > 0:
                transformed_G.edges[u, v]['weight'] = 1.0 / num_outgoing  # Evenly distribute

    for node in transformed_G.nodes():
        out_sum = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_sum, 1.0, atol=1e-3):
            print(f"⚠️ Node {node} outgoing sum = {out_sum}, expected 1")

    return transformed_G  # RETURN as-is, no second normalization!


def apply_no_loops_method(G):
    """
    Apply the no-loops method to a weighted directed graph.

    Parameters:
    G (networkx.DiGraph): Input weighted directed graph. Edges should have 'weight' attribute.

    Returns:
    networkx.DiGraph: Transformed graph with reversed edges and normalized weights
    """
    # Create a new directed graph
    transformed_G = G.reverse(copy=True)  # Reverse edges

    for u, v, data in transformed_G.edges(data=True):
        in_edges = G.in_edges(v, data=True)
        win = sum(data['weight'] for _, _, data in in_edges)  # Compute win(vj)

        if win > 0:
            transformed_G.edges[u, v]['weight'] /= win
        else:
            print(f"Warning: Node {v} has zero in-degree (win = 0). Assigning uniform distribution.")
            num_outgoing = len(list(transformed_G.out_edges(v)))
            if num_outgoing > 0:
                transformed_G.edges[u, v]['weight'] = 1.0 / num_outgoing  # Evenly distribute

    # **Final Check: Normalize Outgoing Weights**
    for node in transformed_G.nodes():
        out_weight_sum = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))

        if out_weight_sum > 0:
            for _, v, data in transformed_G.out_edges(node, data=True):
                data['weight'] /= out_weight_sum  # Normalize outgoing edges

    return transformed_G




def verify_no_loops_transformation(G, transformed_G):
    """
    Verify that the transformation was done correctly:
    1. All original edges are reversed and normalized.
    2. No self-loops exist.
    3. All outgoing probabilities sum to 1.
    """
    # Check for self-loops
    if any(transformed_G.has_edge(node, node) for node in transformed_G.nodes()):
        print("Verification failed: self-loops found in No-Loops method.")
        return False

    # Check if outgoing probabilities sum to 1
    for node in transformed_G.nodes():
        out_weights = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_weights, 1.0, atol=1e-3):
            print(f"Verification failed: Node {node} has outgoing sum {out_weights}, expected 1.")
            print(f"Outgoing edges for {node}: {list(transformed_G.out_edges(node, data=True))}")
            return False

    return True


def Max_weight_arborescence(G_orig:nx.DiGraph):
    # maximum weight arborescence (from the Italy paper: "contrasting the spread of
    # misinformation in online social networks" by Amoruso at. al. 2020)
    max_arbo = nx.maximum_spanning_arborescence(G_orig, attr='weight')
    max_weight_arbo_dict ={}
    # The root of the arborescence is the unique node that has no incoming edges. (i.e. has in-degree of 0)
    for node in max_arbo:
        if max_arbo.in_degree(node) == 0:
            max_weight_arbo_dict[node] = 1
        else:
            max_weight_arbo_dict[node] = 0
    node_dict = max_weight_arbo_dict
    return node_dict



# def calc_stationary_distribution(G, num_steps=1):
#     """
#     returns the stationary distribution of a markov chain network.
#     The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
#      Stationary Distribution of a Markov Chain.)
#     :param G: a nx.DiGraph that is a Markov chain
#     :return: a dict with the pairs-> (node:probability) for every node in G.
#     """
#     # print("len(G.nodes):",len(G.nodes))
#
#     # mat = nx.to_numpy_array(G)
#     # assert(checkMarkov(mat))
#     # evals, evecs = np.linalg.eig(mat.T)
#     # evec1 = evecs[:, np.isclose(evals, 1)]
#     # stationary_distribution = {}
#     #
#     # if (num_steps > 1):
#     #     stationary_distribution = random_walk(G, num_steps)
#     #     return stationary_distribution
#     #
#     # if True in np.isclose(evals, 1):
#     #     evec1 = evec1[:, 0]
#     #     stationary = evec1 / evec1.sum()
#     #     stationary = stationary.real
#     #     stationary = np.array(stationary)
#     #
#     #     node_names = []
#     #     for n in list(G.nodes()):
#     #         node_names.append(n)
#     #     for n in range(len(node_names)):
#     #         # stationary_distribution.update({node_names[n]:stationary[n]})
#     #         stationary_distribution[node_names[n]] = stationary[n]
#     #
#     # else:
#     #     print("Error in computing the stationary distribution.......")
#     #     print("True in np.isclose(evals, 1): ",True in np.isclose(evals, 1))
#     # return stationary_distribution
#
#     mat = nx.to_numpy_array(G, weight="weight")
#     evals, evecs = np.linalg.eig(mat.T)
#     stationary_distribution = {}
#
#     if not np.isclose(evals, 1).any():
#         print("Error: No eigenvalue equals 1.")
#         return {}
#
#     evec1 = evecs[:, np.isclose(evals, 1)][:, 0].real
#     stationary = evec1 / evec1.sum()
#     stationary = stationary.real
#
#     node_names = list(G.nodes())
#     for i, node in enumerate(node_names):
#         stationary_distribution[node] = stationary[i]
#
#     return stationary_distribution

#this from claude
# def calc_stationary_distribution(G, weight_key="weight"):
#     """
#     Returns the stationary distribution of a Markov chain network.
#     Uses eigendecomposition to find the left eigenvector corresponding to eigenvalue 1.
#
#     Parameters:
#     -----------
#     G : nx.DiGraph
#         A directed graph representing a Markov chain where edge weights are transition probabilities
#     weight_key : str, optional
#         The edge attribute used as the weight (default: "weight")
#
#     Returns:
#     --------
#     dict
#         Dictionary mapping nodes to their stationary probabilities
#     """
#     # Get the transition matrix with rows representing source states
#     nodes = list(G.nodes())
#     n = len(nodes)
#     node_to_idx = {node: i for i, node in enumerate(nodes)}
#
#     # Create the transition matrix
#     P = np.zeros((n, n))
#     for u, v, data in G.edges(data=True):
#         i, j = node_to_idx[u], node_to_idx[v]
#         P[i, j] = data.get(weight_key, 0.0)
#
#     # Verify that the matrix is stochastic (rows sum to 1)
#     row_sums = np.sum(P, axis=1)
#     if not np.allclose(row_sums, np.ones(n), rtol=1e-5, atol=1e-8):
#         print("Warning: Transition matrix is not stochastic. Row sums:", row_sums)
#         # Normalize the rows to ensure a stochastic matrix
#         for i in range(n):
#             if row_sums[i] > 0:
#                 P[i, :] /= row_sums[i]
#
#     # Find the left eigenvector corresponding to eigenvalue 1
#     # For a left eigenvector, we use P^T
#     evals, evecs = np.linalg.eig(P.T)
#
#     # Find eigenvalue closest to 1
#     idx = np.argmin(np.abs(evals - 1.0))
#     if not np.isclose(evals[idx], 1.0, rtol=1e-5, atol=1e-8):
#         print(f"Warning: No eigenvalue very close to 1. Closest is {evals[idx]}")
#
#     # Get the eigenvector and ensure it's normalized
#     stationary = evecs[:, idx].real
#     if np.sum(stationary) <= 0:
#         stationary = np.abs(stationary)  # Ensure positive values
#
#     stationary = stationary / np.sum(stationary)  # Normalize to sum to 1
#
#     # Create the result dictionary
#     stationary_distribution = {node: stationary[i] for i, node in enumerate(nodes)}
#
#     return stationary_distribution


def calc_stationary_distribution(G: nx.DiGraph):
    """
    retruns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: an nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """
    # print("len(G.nodes):",len(G.nodes))
    mat = nx.to_numpy_array(G)
    assert(checkMarkov(mat))
    evals, evecs = np.linalg.eig(mat.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    ret_dict = {}
    if True in np.isclose(evals, 1):
        evec1 = evec1[:, 0]
        stationary = evec1 / evec1.sum()
        stationary = stationary.real
        stationary = np.array(stationary)

        node_names = []
        for n in list(G.nodes()):
            node_names.append(n)
        for n in range(len(node_names)):
            # ret_dict.update({node_names[n]:stationary[n]})
            ret_dict[node_names[n]] = stationary[n]
    else:
        print("Error in computing the stationary distribution.......")
        print("True in np.isclose(evals, 1): ",True in np.isclose(evals, 1))
    return ret_dict

def calc_normalized_stationary_distribution(G, G_orignal, num_steps=1):
    """
    returns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: a nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """
    # print("len(G.nodes):",len(G.nodes))

    # mat = nx.to_numpy_array(G)
    # assert(checkMarkov(mat))
    # evals, evecs = np.linalg.eig(mat.T)
    # evec1 = evecs[:, np.isclose(evals, 1)]
    # stationary_distribution = {}
    #
    # if (num_steps > 1):
    #     stationary_distribution = random_walk(G, num_steps)
    #     return stationary_distribution
    #
    # if True in np.isclose(evals, 1):
    #     evec1 = evec1[:, 0]
    #     stationary = evec1 / evec1.sum()
    #     stationary = stationary.real
    #     stationary = np.array(stationary)
    #
    #     node_names = []
    #     for n in list(G.nodes()):
    #         node_names.append(n)
    #     for n in range(len(node_names)):
    #         # stationary_distribution.update({node_names[n]:stationary[n]})
    #         stationary_distribution[node_names[n]] = stationary[n]
    #
    # else:
    #     print("Error in computing the stationary distribution.......")
    #     print("True in np.isclose(evals, 1): ",True in np.isclose(evals, 1))

    mat = nx.to_numpy_array(G, weight="weight")
    evals, evecs = np.linalg.eig(mat.T)
    stationary_distribution = {}

    if not np.isclose(evals, 1).any():
        print("Error: No eigenvalue equals 1.")
        return {}

    evec1 = evecs[:, np.isclose(evals, 1)][:, 0].real
    stationary = evec1 / evec1.sum()
    stationary = stationary.real

    node_names = list(G.nodes())
    for i, node in enumerate(node_names):
        stationary_distribution[node] = stationary[i]

    normalized_distribution = {}
    for i, node in enumerate(G.nodes()):
        win = sum(data['weight'] for _, _, data in G_orignal.in_edges(node, data=True))
        if win > 0:
            normalized_distribution[node] = stationary_distribution[node] / win

    return normalized_distribution

    # normalized_stationary_distribution = {}
    # for node in stationary_distribution.keys():
    #     in_edges = G_orignal.in_edges(node, data=True)
    #     win = sum(data['weight'] for _, _, data in in_edges)
    #     normalized_stationary_distribution.update({node: (stationary_distribution[node] / win) })
    # stationary_distribution = normalized_stationary_distribution

    # # Final normalization
    # total = sum(normalized_stationary_distribution.values())
    # for node in normalized_stationary_distribution:
    #     normalized_stationary_distribution[node] /= total

    # return normalized_stationary_distribution



def find_most_probable_source(G,num_steps=1):
    """
    Find the most probable source in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    # Calculate the stationary distribution
    stationary_distribution = calc_stationary_distribution(G)

    # Find the node with the maximum stationary probability
    if not stationary_distribution:
        return -1, -1
    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    max_prob = stationary_distribution[most_probable_node]

    return most_probable_node, max_prob

def find_most_probable_source_no_loop(G, G_orignal, num_steps=1):
    """
    Find the most probable source in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    # Calculate the stationary distribution
    stationary_distribution = calc_normalized_stationary_distribution(G, G_orignal,num_steps)

    # Find the node with the maximum stationary probability
    if not stationary_distribution:
        return -1, -1
    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    max_prob = stationary_distribution[most_probable_node]

    return most_probable_node, max_prob

def find_most_probable_source_no_loops_new(G, G_originanl, num_steps = 1):
    normalized_stationary_distribution = {}
    stationary_distribution = calc_stationary_distribution(G)
    if not stationary_distribution:
        return -1, -1

    win_prob = find_Win_prob(G_originanl)
    # print("Win probs: ", win_prob)
    for node in stationary_distribution:
        if win_prob[node] == 0:
            normalized_stationary_distribution[node] = 0  # handling division by 0
            continue
        normalized_stationary_distribution[node] =stationary_distribution[node]/win_prob[node]

    # print("Stationary dist before normalization: " , stationary_distribution)
    print("Stationary dist for no loops after normalization: ", normalized_stationary_distribution)
    most_probable_node = max(normalized_stationary_distribution, key=normalized_stationary_distribution.get)
    max_prob = normalized_stationary_distribution[most_probable_node]

    return most_probable_node, max_prob




def find_Win_prob(G):
    win_probs = {}
    for node in G.nodes:
        # Get all incoming edges to this node in the original graph
        incoming_edges = G.in_edges(node, data=True)
        total_weight = sum(attr.get('weight', 1) for _, _, attr in incoming_edges)
        win_probs[node] = total_weight
    return win_probs

def find_top_three(G, num_steps = 1):
    """
    Find the 3 most probable source nodes in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source nodes and their stationary distribution values.
    """
    # Calculate the stationary distribution
    stationary_distribution = calc_stationary_distribution(G, num_steps)

    # Find the node with the maximum stationary probability
    if not stationary_distribution:
        return -1, -1

    # Get the 3 most probable nodes along with their probabilities
    top_3_nodes = heapq.nlargest(3, stationary_distribution.items(), key=lambda x: x[1])

    # Extract the nodes and their probabilities
    top_3_nodes_list = [(node, prob) for node, prob in top_3_nodes]

    return top_3_nodes, top_3_nodes_list


def is_most_probable_near_source(G, source_node):
    stationary_distribution = calc_stationary_distribution(G)

    if not stationary_distribution:
        return False

    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    shortest_path_length = nx.shortest_path_length(G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3


def is_most_probable_near_source_no_loop(G, G_original, source_node):
    stationary_distribution = calc_normalized_stationary_distribution(G, G_original)

    if not stationary_distribution:
        return False

    most_probable_node = max(stationary_distribution, key=stationary_distribution.get)
    shortest_path_length = nx.shortest_path_length(G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3

def is_most_probable_near_source_max_arbo(Max_weight_arborescence_G, source_node):

    most_probable_node = max(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get)
    shortest_path_length = nx.shortest_path_length(Max_weight_arborescence_G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3



def checkMarkov(m):
    """
    a function to assert that the given matrix is a Markov chain.
    (the function checks if the sum of each row is 1.)
    :param m: a matrix
    :return: bool value
    """
    # for i in range(0 , len(m)):
    #
    #     # Find sum of current row
    #     sm = 0
    #     for j in range(0 , len(m[ i ])):
    #         sm = sm + m[ i ][ j ]
    #
    #     if (sm - 1>0.001):
    #         print("sum of line is:", sm)
    #         return False
    # return True

    return np.all(np.isclose(np.sum(m, axis=1), 1))

def random_walk(G:nx.DiGraph, num_steps):
    '''
    this function performs a random walk estimation of a stationary distribution of a markov chain
    :param G: a DiGraph representing the network
    :param num_steps: number of steps of the random walk
    :return: a dict where  {node: number of times the random walk visited node}
    '''
    if not G.nodes():
        return {}
    random_start = random.choice(list(G.nodes()))
    nodes_on_path = [random_start]
    curr = random_start

    for step in range(num_steps):
        #create the list of the neighbors of curr:
        neighbors_list =[friend for friend in G.neighbors(curr)]

        #create the list of weights: for every neighbor v, we use the weight of the edge (curr,v)
        weights_list = []
        for neig in neighbors_list:
            weights_list.append(G.edges[curr,neig]['weight'])

        if len(neighbors_list) > 0:
            #random.choices returns a list of k=1 selections from neigbors_list, according to the weights_list:
            curr = random.choices(neighbors_list , weights=weights_list , k=1)[0]
            nodes_on_path.append(curr)

    #create the dict that summarise the number of visits to each node:
    ret_dict = {}
    for node in G.nodes:
        ret_dict[node] = nodes_on_path.count(node)
    return ret_dict