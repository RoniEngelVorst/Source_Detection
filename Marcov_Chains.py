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
        # Check if outgoing probabilities sum to 1 (within numerical precision)
        out_weights = sum(data['weight'] for _, _, data in transformed_G.out_edges(node, data=True))
        if not np.isclose(out_weights, 1.0, atol=1e-3):  # Absolute tolerance
            print("out weights is: ", out_weights)
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
    transformed_G = G.reverse(copy=True)

    # Reverse edges and normalize by win(v_j)
    for u, v in transformed_G.edges:
        in_edges = G.in_edges(v, data=True)
        win = sum(data['weight'] for _, _, data in in_edges)
        if win > 0:
            transformed_G.edges[u, v]['weight'] /= win

    return transformed_G


    # win = {}
    # for node in G.nodes():
    #     in_edges = G.in_edges(node, data=True)
    #     win[node] = sum(data['weight'] for _, _, data in in_edges)
    #     # if win[node] == 0:
    #     #     raise ValueError(f"Node {node} has no incoming edges")
    #
    # # Convert edges with normalized weights
    # for u, v, data in G.edges(data=True):
    #     # Create reversed edge with normalized weight qji = pij/win(vj)
    #     normalized_weight = data['weight'] / win[v]
    #     transformed_G.add_edge(v, u, weight=normalized_weight)

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
        print("there is a self loop")
        return False

    # Check if all edges are properly reversed and normalized
    for node in transformed_G.nodes():
        out_edges = transformed_G.out_edges(node, data=True)
        out_weights = sum(data['weight'] for _, _, data in out_edges)
        if not np.isclose(out_weights, 1.0, rtol=1e-3):
            print("the sum of weights is not 1")
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



def calc_stationary_distribution(G, num_steps=1):
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
    # return stationary_distribution

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

    return stationary_distribution


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
    stationary_distribution = calc_stationary_distribution(G,num_steps)

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

