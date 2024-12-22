import networkx as nx
import random
import matplotlib.pyplot as plt



def random_graph_generator(numberOfNodes, ProbOfAnEdge, maxProbForDiffusion):
    """
       A function that generates a random graph based on the given parameters:
       - numberOfNodes: the number of nodes in the graph.
       - ProbOfAnEdge: the probability of having an edge between any two nodes.
       - maxProbForDiffusion: the maximum probability for edge weights (for diffusion probabilities).

       The function creates a directed graph, and assigns random diffusion probabilities to each edge.

       :param numberOfNodes: number of nodes in the graph
       :param ProbOfAnEdge: probability for edge creation between any pair of nodes
       :param maxProbForDiffusion: maximum diffusion probability for the edges

       :return: the resulting directed random graph as a nx.Graph object
       """
    # Create a random directed graph using the specified edge probability
    G = nx.fast_gnp_random_graph(numberOfNodes, ProbOfAnEdge, directed=True)

    # Assign random weights to the edges based on the maximum diffusion probability
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = random.random() * maxProbForDiffusion

    return G


def visualize_graph(G):
    """
    Visualizes the graph with nodes, edges, and optionally edge weights.
    """
    pos = nx.spring_layout(G)  # Spring layout

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.7, edge_color='gray')

    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='black')

    # Only draw edge labels if the graph is small enough
    if len(G.edges()) < 100:  # Example condition, adjust as needed
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    plt.title("Graph Visualization with Edge Weights")
    plt.axis('off')
    plt.show()


def visualize_subgraph(G, max_nodes=100):
    """
    Visualizes a subgraph of the graph with nodes, edges, and edge weights.
    Displays edge weights with 3 decimal places, regardless of the number of edges.
    """
    # Sample a subgraph with a maximum number of nodes
    subgraph_nodes = list(G.nodes())[:max_nodes]  # Select the first 'max_nodes' nodes
    subgraph = G.subgraph(subgraph_nodes)

    pos = nx.spring_layout(subgraph, seed=42)  # Using a fixed seed for reproducibility

    plt.figure(figsize=(12, 10))  # Increase figure size for better visibility
    nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(subgraph, pos, width=1.0, alpha=0.7, edge_color='gray')

    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold', font_color='black')

    # Get edge weights (probabilities) and format them with 3 decimal places
    edge_labels = nx.get_edge_attributes(subgraph, 'weight')

    if not edge_labels:  # Debugging if there are no edge labels
        print("No edge weights found in the subgraph.")

    formatted_edge_labels = {edge: f"{weight:.3f}" for edge, weight in edge_labels.items()}

    # Draw edge labels (with formatted probabilities) without checking the number of edges
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=formatted_edge_labels, font_size=8, font_color='red')

    plt.title("Graph Visualization (Subgraph)")
    plt.axis('off')
    plt.show()



def visualize_large_graph(G, max_nodes=1000, layout='spring'):
    """
    Visualizes a large graph by displaying only a subgraph with a maximum number of nodes.
    Uses either a spring layout or spectral layout for large graphs.
    """
    # Convert G.nodes() to a list before using random.sample()
    subgraph_nodes = random.sample(list(G.nodes()), min(max_nodes, len(G.nodes())))  # Randomly sample nodes
    subgraph = G.subgraph(subgraph_nodes)

    if layout == 'spring':
        pos = nx.spring_layout(subgraph, seed=42, k=0.15, iterations=20)
    elif layout == 'spectral':
        pos = nx.spectral_layout(subgraph)  # Faster layout for large graphs
    else:
        pos = nx.circular_layout(subgraph)  # Use circular layout for simplicity

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(subgraph, pos, width=0.5, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color='black')

    plt.title(f"Graph Visualization (Subgraph - {len(subgraph.nodes())} nodes)")
    plt.axis('off')
    plt.show()
