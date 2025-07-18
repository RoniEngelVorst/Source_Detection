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


def reverse_and_normalize_weights_no_loop(G_original):  # added by hadar!
    """
    Constructs a reversed graph using the 'no-loops' method.
    Edges are reversed, and weights are divided by the in-degree of the original source.
    """
    reversed_G = nx.DiGraph()
    in_weights = {}

    # First compute weighted in-degree for each node
    for u, v, data in G_original.edges(data=True):
        weight = data.get('weight', 1.0)
        in_weights[v] = in_weights.get(v, 0.0) + weight

    # Now build the reversed graph with normalized edge weights
    for u, v, data in G_original.edges(data=True):
        weight = data.get('weight', 1.0)
        if in_weights[v] > 0:
            normalized_weight = weight / in_weights[v]
            reversed_G.add_edge(v, u, weight=normalized_weight)

    return reversed_G, in_weights


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

def roni_self_loops(G: nx.DiGraph) -> nx.DiGraph:
    loop_reversed_graph = G.reverse(copy=True)

    # find the maximum in degree (of the original graph)
    max_in_degree = 0
    for node in G:
        node_in_degree = G.in_degree(node, weight="weight")
        if node_in_degree > max_in_degree:
            max_in_degree = node_in_degree

    # divide each edge's weight in max_in_degree
    for u, v, data in loop_reversed_graph.edges(data=True):
        if "weight" in data:
            data["weight"] /= max_in_degree
        else:
            data["weight"] = 1.0 / max_in_degree

    win_prob = find_Win_prob(G)
    # calculating the sum of the weight of the out edges for each node and creating the self loop with the difference
    for node in G.nodes():
        sum_prob = win_prob[node]
        missing = (max_in_degree - sum_prob) / max_in_degree
        loop_reversed_graph.add_edge(node, node, weight=missing)

    # final normalization in order to sum to 1
    for node in loop_reversed_graph.nodes:
        out_edges = list(loop_reversed_graph.out_edges(node, data=True))
        total = sum(data["weight"] for _, _, data in out_edges)
        for u, v, data in out_edges:
            data["weight"] /= total

    return loop_reversed_graph


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
        if not np.isclose(out_weights, 1.0, atol= 0.001):
            print(f"Verification failed: Node {node} has outgoing sum {out_weights}, expected 1.")
            print(f"Outgoing edges for {node}: {list(transformed_G.out_edges(node, data=True))}")
            return False

    return True


def verify_no_loops_transformation(G, transformed_G):
    """
    Verify that the transformation was done correctly:
    1. All original edges are reversed and normalized.
    2. All outgoing probabilities sum to 1.
    """

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


def Approx_max_weight_arborescence(G_orig: nx.DiGraph):  # added by hadar
    """
    Approximate version of max-weight arborescence.
    For each node, keeps the max incoming edge.
    The root (no incoming edge) is marked 1, others 0.
    """
    max_weight_arbo_dict = {}

    for node in G_orig.nodes:
        in_edges = G_orig.in_edges(node, data=True)
        if not in_edges:
            max_weight_arbo_dict[node] = 1  # root
        else:
            _ = max(in_edges, key=lambda x: x[2].get('weight', 0))
            max_weight_arbo_dict[node] = 0

    return max_weight_arbo_dict



def calc_stationary_distribution(G: nx.DiGraph):
    """
    retruns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: an nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """
    mat = nx.to_numpy_array(G)
    assert(checkMarkov(mat))
    evals, evecs = np.linalg.eig(mat.T)
    evec1 = evecs[:, np.isclose(evals, 1, atol=0.001)]
    ret_dict = {}
    if True in np.isclose(evals, 1, atol=0.001):
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
        print("True in np.isclose(evals, 1, atol=0.001): ",True in np.isclose(evals, 1, atol=0.001))
    return ret_dict



def calc_normalized_stationary_distribution(G, G_orignal, num_steps=1):
    """
    returns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: a nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """

    mat = nx.to_numpy_array(G, weight="weight")
    evals, evecs = np.linalg.eig(mat.T)
    stationary_distribution = {}

    if not np.isclose(evals, 1, atol=0.01).any():
        print("Error: No eigenvalue close enough to 1.")
        return {}

    evec1 = evecs[:, np.isclose(evals, 1, atol=0.01)][:, 0].real
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


def yael_stationary_distribution(G: nx.DiGraph):
    """
    retruns the stationary distribution of a markov chain network.
    The basic logic here is finding the eigen vector that matches to the eigen value =1. (This is a main property of the
     Stationary Distribution of a Markov Chain.)
    :param G: an nx.DiGraph that is a Markov chain
    :return: a dict with the pairs-> (node:probability) for every node in G.
    """
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

def find_most_probable_source_no_loop(G, G_original, num_steps=1):
    """
    Find the most probable source in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    normalized_stationary_distribution = {}
    stationary_distribution = calc_stationary_distribution(G)
    # stationary_distribution = yael_stationary_distribution(G) #worked the same as self loops

    if not stationary_distribution:
        return -1, -1

    win_prob = find_Win_prob(G_original)
    # print("Win probs: ", win_prob)
    for node in stationary_distribution:
        if win_prob[node] == 0:
            normalized_stationary_distribution[node] = 0  # handling division by 0
            continue
        normalized_stationary_distribution[node] = stationary_distribution[node] / win_prob[node]


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

def find_top_three(G, num_steps=1):
    """
    Returns the top 3 nodes based on stationary distribution.
    """
    stationary_distribution = calc_stationary_distribution(G)

    if not stationary_distribution:
        return []

    # Just return the nodes, sorted by probability
    top_3_nodes = heapq.nlargest(3, stationary_distribution.items(), key=lambda x: x[1])
    return [node for node, _ in top_3_nodes]



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

def is_most_probable_near_source_max_arbo(Max_weight_arborescence_G, G, source_node):

    most_probable_node = max(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get)
    shortest_path_length = nx.shortest_path_length(G, source=source_node, target=most_probable_node)

    return shortest_path_length <= 3


def checkMarkov(m):
    """
    Check if the given matrix is a valid Markov chain transition matrix,
    allowing a tolerance of 0.01 for row sums.

    Parameters:
    m (numpy.ndarray): The matrix to check.

    Returns:
    bool: True if each row sums approximately to 1 (±0.01), False otherwise.
    """
    return np.all(np.isclose(np.sum(m, axis=1), 1.0, atol=0.01))

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




# K-Sources Methods


def find_K_most_probable_sources(G,k, num_steps=1):
    """
    Find the k most probable sources in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    # Calculate the stationary distribution
    stationary_distribution = calc_stationary_distribution(G)

    # Find the nodes with the maximum stationary probability
    if not stationary_distribution:
        return -1, -1

    top_k_nodes = sorted(stationary_distribution, key=stationary_distribution.get, reverse=True)[:k]
    top_k_probs = [stationary_distribution[node] for node in top_k_nodes]

    return top_k_nodes, top_k_probs


def find_K_most_probable_sources_no_loop(G, G_original, k, num_steps=1):
    """
    Find the k most probable sources in a Markov chain represented by a NetworkX DiGraph,
    based on the stationary distribution.

    Args:
    G (networkx.DiGraph): The directed graph representing the Markov chain.

    Returns:
    tuple: The most probable source node and its stationary distribution value.
    """
    normalized_stationary_distribution = {}
    stationary_distribution = calc_stationary_distribution(G)
    if not stationary_distribution:
        return -1, -1

    win_prob = find_Win_prob(G_original)
    # print("Win probs: ", win_prob)
    for node in stationary_distribution:
        if win_prob[node] == 0:
            normalized_stationary_distribution[node] = 0  # handling division by 0
            continue
        normalized_stationary_distribution[node] = stationary_distribution[node] / win_prob[node]


    top_k_nodes = sorted(normalized_stationary_distribution, key=normalized_stationary_distribution.get, reverse=True)[ :k]
    top_k_probs = [normalized_stationary_distribution[node] for node in top_k_nodes]

    return top_k_nodes, top_k_probs

# Evaluation methods for k sources

# A function that returns the percentage of success - from the real k sources, how many are in our prediction sources - Recall
def percent_exact_matches(real_sources, estimated_sources):
    return len(set(real_sources) & set(estimated_sources)) / len(real_sources)

# From our predicted sources how many of them are real ones
def precision_of_estimation(real_sources, estimated_sources):
    return len(set(real_sources) & set(estimated_sources)) / len(estimated_sources)

def count_sources_within_distance_k(G, real_sources, estimated_sources, max_distance=3):
    count = 0
    for real_node in real_sources:
        if any(
            nx.has_path(G, real_node, est_node) and
            nx.shortest_path_length(G, real_node, est_node) <= max_distance
            for est_node in estimated_sources
        ):
            count += 1
    return count

def percent_sources_within_distance_k(G, real_sources, estimated_sources, max_distance=3):
    return count_sources_within_distance_k(G, real_sources, estimated_sources, max_distance) / len(real_sources)


