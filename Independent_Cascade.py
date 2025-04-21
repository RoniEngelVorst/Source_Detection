import random
import networkx as nx
import matplotlib.pyplot as plt



# def simulate_ic_model(G, source_node, max_iterations=500):
#     """
#     Simulates the Independent Cascade (IC) model on the graph with a given source node.
#
#     :param G: The graph on which the simulation is run.
#     :param source_node: The initial node that will be infected.
#     :param max_iterations: Maximum number of iterations before stopping the simulation.
#
#     :return: A set of infected nodes.
#     """
#     # Step 1: Initialize all nodes as not infected
#     infected = set([source_node])  # Set of infected nodes (starting with the source)
#     new_infected = set([source_node])  # Set of nodes to try infecting in the next iteration
#
#     # Step 2: Simulate the spread of the infection
#     iterations = 0
#     while new_infected and iterations < max_iterations:
#         next_infected = set()
#         for node in new_infected:
#             # Try to infect neighbors of the currently infected node
#             for neighbor in G.neighbors(node):
#                 if neighbor not in infected:
#                     # Check if the neighbor gets infected based on the edge weight (probability)
#                     infection_probability = G[node][neighbor].get('weight', 0)
#                     if random.random() < infection_probability:
#                         next_infected.add(neighbor)
#         infected.update(next_infected)
#         new_infected = next_infected
#         iterations += 1
#
#     return infected
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



def Atag_calc(G):
    """
    :param G: a graph, induced on the active nodes
    :return: the set of possible sources. (i.e. this function deletes from the graph all the nodes that can't reach all
    the other nodes of the active set.)
    """
    Atag=[]
    for i in G.nodes:
        #a test that ensures that all the nodes of the graph are reached from i by a directed path.
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

        #a logic text, that all the nodes of G are in the list of visited, if it returns true, then i is inserted to Atag:
        visit_all_active = True
        for j in G.nodes:
            visit_all_active = visit_all_active & (j in visited)
        if visit_all_active:
            Atag.append(i)

    return Atag


def Atag_calc_infected(G, infected_set):
    """
    :param G: Directed graph
    :param infected_set: Set of infected nodes
    :return: Set of possible source nodes from the infected set
    """
    possible_sources = set()

    for node in infected_set:
        visited = set()
        queue = [node]

        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                # Only consider neighbors within the infected set
                queue.extend(neighbor for neighbor in G.neighbors(curr) if neighbor in infected_set and neighbor not in visited)

        # Check if this node can reach all infected nodes
        if infected_set.issubset(visited):
            possible_sources.add(node)

    return possible_sources

def create_induced_subgraph(G, nodes):
    """
    Creates the induced subgraph of G using the specified nodes.

    :param G: The original graph
    :param nodes: A set or list of nodes to include in the induced subgraph
    :return: The induced subgraph
    """
    return G.subgraph(nodes).copy()