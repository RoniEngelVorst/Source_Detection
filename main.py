from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *


def main():
    # Create a graph
    # G = nx.DiGraph()
    # G.add_edges_from([
    #     (1, 2), (2, 3), (3, 4), (4, 5), (5, 1), (3, 5)
    # ])

    G1 = random_graph_generator(100,0.7,0.0416)
    print("starting")
    source_node = random.choice(list(G1.nodes()))
    print("the source node is: ", source_node)
    infected_nodes = simulate_ic_model(G1, source_node, max_iterations=len(G1.nodes))
    print("running the ic model")
    possible_sources = Atag_calc_infected(G1, infected_nodes)
    print(f"Posssible sources: {possible_sources}")
    print(f"Number of possible sources of infection: {len(possible_sources)}")
    induced_graph = create_induced_subgraph(G1, possible_sources)

    visualize_graph(induced_graph)
    self_loops_G = apply_self_loop_method(induced_graph)
    visualize_graph(self_loops_G)
    print(f"did the self loops method work? {verify_self_loops_transformation(induced_graph, self_loops_G)}")

    pi = calc_stationary_distribution(self_loops_G)
    print("Stationary distribution:", pi)

    # # Define the infected set
    # infected_set = {1, 3, 5}
    #
    # # Find possible sources
    # possible_sources = Atag_calc_infected(G, infected_set)
    # print("Possible sources:", possible_sources)

    # Example usage
    # G = nx.DiGraph()
    # G.add_edge(1, 2, weight=0.5)
    # G.add_edge(2, 4, weight=0.2)
    # G.add_edge(4, 1, weight=0.8)
    # G.add_edge(1, 3, weight=0.1)
    # G.add_edge(3, 2, weight=0.3)
    # G.add_edge(4, 1, weight=0.8)
    # G.add_edge(2, 5, weight=0.3)
    # G.add_edge(4, 1, weight=0.2)
    # G.add_edge(3, 4, weight=0.6)
    # G.add_edge(2, 3, weight=0.3)
    # G.add_edge(1, 2, weight=0.1)
    # G.add_edge(4, 2, weight=0.4)

    reversed_G = reverse_and_normalize_weights(induced_graph)
    visualize_graph(reversed_G)

    # visualize_graph(G)

    # reversed_G = reverse_and_normalize_weights(G)
    # print("Reversed edges with normalized weights:")
    # for u, v, data in reversed_G.edges(data=True):
    #     print(f"Edge ({u} -> {v}): {data}")
    #
    # visualize_graph(reversed_G)

    # self_loops_G = apply_self_loop_method(G)
    # visualize_graph(self_loops_G)
    # print(f"did the self loops method work? {verify_self_loops_transformation(G, self_loops_G)}")

    # visualize_graph(induced_graph)
    # no_loops_G = apply_no_loops_method(induced_graph)
    # visualize_graph(no_loops_G)
    # print(f"did the no loops method work? {verify_no_loops_transformation(induced_graph, no_loops_G)}")

    # # Generate the graph
    # G = random_graph_generator(100, 0.1, 0.15)
    # source_node = random.choice(list(G.nodes()))  # Choose a random source node
    # infected_nodes = simulate_ic_model(G, source_node, max_iterations=len(G.nodes))
    #
    # visualize_large_graph(G,500)
    # visualize_large_graph(create_induced_subgraph(G, infected_nodes))
    #
    # print(f"Initial source node: {source_node}")
    # print(f"Number of infected nodes: {len(infected_nodes)}")
    # print(f"Infected nodes: {infected_nodes}")
    #
    # # Find possible sources among the infected nodes
    # possible_sources = Atag_calc_infected(G, infected_nodes)
    # print(f"Posssible sources: {possible_sources}")
    # print(f"Number of possible sources of infection: {len(possible_sources)}")
    #
    # visualize_large_graph(create_induced_subgraph(G, possible_sources))




if __name__ == '__main__':
    main()