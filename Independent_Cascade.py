import random
from collections import deque

import networkx as nx
import matplotlib.pyplot as plt


def simulate_ic_model(G, source_node, max_iterations=500, seed=None):
    """
    Simulates the Independent Cascade (IC) model on the graph with a given source node.

    :param G: The graph on which the simulation is run (must have 'weight' attributes on edges).
    :param source_node: The initial node that will be infected.
    :param max_iterations: Maximum number of iterations before stopping the simulation.
    :param seed: Optional random seed for reproducibility.

    :return: A set of infected nodes.
    """
    if seed is not None:
        random.seed(seed)

    # Validate edge weights
    assert all('weight' in G[u][v] for u, v in G.edges), "All edges must have a 'weight' attribute!"
    assert all(0 <= G[u][v]['weight'] <= 1 for u, v in G.edges), "Edge weights must be in [0, 1]!"

    # Step 1: Initialize infection state
    infected = set([source_node])  # Nodes that are infected
    new_infected = set([source_node])  # Nodes to attempt to infect in the next iteration

    # Step 2: Simulate the spread of the infection
    iterations = 0
    while new_infected and iterations < max_iterations:
        next_infected = set()
        for node in new_infected:
            for neighbor in G.neighbors(node):  # Only considers outgoing edges
                if neighbor not in infected:
                    # Infection probability
                    infection_probability = G[node][neighbor]['weight']
                    if random.random() < infection_probability:
                        next_infected.add(neighbor)
        infected.update(next_infected)
        new_infected = next_infected
        iterations += 1

    return infected

def visualize_infection(G, infected_nodes):
    color_map = ['red' if node in infected_nodes else 'skyblue' for node in G.nodes]
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=color_map)
    plt.show()



def  Atag_calc(G):
    """
    :param G: a graph, induced on the active nodes
    :return: the set of possible sources. (i.e. this function deletes from the graph all the nodes that can't reach all
    the other nodes of the active set.)
    """
    Atag=[]
    for i in G.nodes:
        # a test that ensures that all the nodes of the graph are reached from i by a directed path.
        visited = []  # List to keep track of visited nodes.
        queue = []  # Initialize a queue
        visited.append(i)
        queue.append(i)
        while len(queue)>0:
            curr = queue.pop(0)
            for neighbour in G.neighbors(curr):
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
        # a logic text, that all the nodes of G are in the list of visited, if it returns true, then i is inserted to Atag:
        visit_all_active = True
        for j in G.nodes:
            visit_all_active = visit_all_active & (j in visited)
        if visit_all_active:
            Atag.append(i)

    return Atag


#
# def Atag_calc_scc(G):
#     """
#     Fast source detection using strongly connected components (SCCs).
#     Only nodes in the largest SCC that spans the graph can be sources.
#     """
#     all_nodes = set(G.nodes)
#     for component in nx.strongly_connected_components(G):
#         if set(component) == all_nodes:
#             return list(component)  # all nodes can reach all others
#     return []  # no node can reach everyone
#
#
# import networkx as nx
#
# def compute_A_tag(graph):
#     """
#     Computes A', the set of nodes in the given subgraph from which
#     there exists a directed path to every other node in the subgraph.
#
#     This corresponds exactly to the definition in the paper:
#     A' = { v in A | for all u in A: there is a directed path from v to u }
#
#     To do this efficiently, we reverse the graph and for each node v,
#     we check whether all other nodes can reach v (which is equivalent).
#     """
#     reversed_graph = graph.reverse(copy=True)
#     print("Graph reversed.")
#     nodes = list(reversed_graph.nodes())
#     A_tag = []
#
#     for v in nodes:
#         # If all other nodes can reach v in the reversed graph,
#         # then in the original graph v reaches all others.
#         if nx.descendants(reversed_graph, v) >= set(nodes) - {v}:
#             A_tag.append(v)
#
#     print("finished computing Atag.")
#     return A_tag
#
# def compute_A_tag_fast(graph):
#     """
#     Efficient approximation of A' for large graphs:
#     Returns all nodes from which the rest are reachable.
#     """
#     print("starting to compute Atag fast")
#     reversed_graph = graph.reverse(copy=True)
#     start = next(iter(reversed_graph.nodes()))
#     reachable = nx.descendants(reversed_graph, start)
#     reachable.add(start)
#     print("finished computing Atag")
#     return list(reachable)
#
#
# def Atag_calc_fast(G):
#     """
#     :param G: a directed graph (DiGraph) induced on active nodes
#     :return: list of possible source nodes — those that can reach all others
#     """
#     all_nodes = set(G.nodes)
#     reversed_G = G.reverse(copy=False)
#
#     possible_sources = []
#     for node in G.nodes:
#         reachable_to_node = nx.descendants(reversed_G, node)
#         reachable_to_node.add(node)  # include self
#
#         if reachable_to_node == all_nodes:
#             possible_sources.append(node)
#
#     return possible_sources
#
#
#
# def Atag_calc_infected(G, infected_set):
#     """
#     :param G: Directed graph
#     :param infected_set: Set of infected nodes
#     :return: Set of possible source nodes from the infected set
#     """
#     possible_sources = set()
#
#     for node in infected_set:
#         visited = set()
#         queue = [node]
#
#         while queue:
#             curr = queue.pop(0)
#             if curr not in visited:
#                 visited.add(curr)
#                 # Only consider neighbors within the infected set
#                 queue.extend(neighbor for neighbor in G.neighbors(curr) if neighbor in infected_set and neighbor not in visited)
#
#         # Check if this node can reach all infected nodes
#         if infected_set.issubset(visited):
#             possible_sources.add(node)
#
#     return possible_sources

def create_induced_subgraph(G, nodes):
    """
    Creates the induced subgraph of G using the specified nodes.

    :param G: The original graph
    :param nodes: A set or list of nodes to include in the induced subgraph
    :return: The induced subgraph
    """
    return G.subgraph(nodes).copy()


# K-Sources Methods

def simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=500, seed=None):
    """
    Simulates the Independent Cascade (IC) model on the graph with a given source node.

    :param G: The graph on which the simulation is run (must have 'weight' attributes on edges).
    :param source_nodes: The initial nodes that will be infected.
    :param max_iterations: Maximum number of iterations before stopping the simulation.
    :param seed: Optional random seed for reproducibility.

    :return: A set of infected nodes.
    """
    if seed is not None:
        random.seed(seed)

    # Validate edge weights
    assert all('weight' in G[u][v] for u, v in G.edges), "All edges must have a 'weight' attribute!"
    assert all(0 <= G[u][v]['weight'] <= 1 for u, v in G.edges), "Edge weights must be in [0, 1]!"

    # Step 1: Initialize infection state
    infected = set(source_nodes)  # Nodes that are infected
    new_infected = set(source_nodes)  # Nodes to attempt to infect in the next iteration

    # Step 2: Simulate the spread of the infection
    iterations = 0
    while new_infected and iterations < max_iterations:
        next_infected = set()
        for node in new_infected:
            for neighbor in G.neighbors(node):  # Only considers outgoing edges
                if neighbor not in infected:
                    # Infection probability
                    infection_probability = G[node][neighbor]['weight']
                    if random.random() < infection_probability:
                        next_infected.add(neighbor)
        infected.update(next_infected)
        new_infected = next_infected
        iterations += 1

    return infected
