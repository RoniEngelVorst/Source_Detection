import networkx as nx
import numpy as np
import time
# my models:
from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *
import random

def Append_to_file(file_name, text):
    print(file_name, ":", text)
    file = open(file_name,"a")
    file.write(text + "\n")
    file.close()


def run_real_graphs():
    begin_time = time.time()

    real_graphs = [
        # (r"real_graphs\advogato\out.advogato", "advogato", " ", 0.257),  # finished running
        # (r"real_graphs\digg-friends\out.digg-friends", "digg", " ", 0.694),  # need to run on server
        # (r"real_graphs\soc-Epinions1\out.soc-Epinions1", "epinion_trust", "\t", 0.299),  # need to run on server
        (r"real_graphs\facebook-wosn-links\out.facebook-wosn-links", "facebook_friendships", " ", 0.157), # need to check
        # (r"real_graphs\ego-gplus\out.ego-gplus", "google_plus", "\t", 1),  # need to run on server
        # (r"real_graphs\slashdot-threads\out.slashdot-threads", "slashdot", " ", 0.726),  # Our fix. need to check
        # (r"real_graphs\ego-twitter\out.ego-twitter", "twitter", "\t", 1),  # need to run on server

        # (r"real_graphs\youtube-links\out.youtube-links" , "youtube_links" , " " , 0.461) ,  # really big. Yael didnt run it without random walks.
        # (r"real_graphs\youtube_friendship\out.com-youtube", "youtube_friendships", " ", 4),  # Not in the table
        # (r"real_graphs\slashdot\out.matrix", "slashdot", " ", 3),  # What Yeal wrote. but it is not in the folder
        # (r"real_graphs\facebook_nips\out.ego-facebook", "facebook_nips", " ", 10), # Not in the article
    ]

    min_size_of_diffusion = 20
    max_size_of_graph = 20000
    # max_diffusions_per_graph = 1000 # check if needed?

    for (path, graph_name, sep_char, P_range) in real_graphs:
        # naive_num_of_successes = 0
        # no_loop_num_of_successes = 0
        # self_loop_num_of_successes = 0
        # max_arbo_num_of_successes = 0

        # Initialize success counters
        # naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        # self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0

        # naive_top3_successes = 0
        no_loop_top3_successes = 0
        # self_loop_top3_successes = 0
        max_arbo_top3_successes = 0

        # naive_near_successes = 0
        no_loop_near_successes = 0
        # self_loop_near_successes = 0
        max_arbo_near_successes = 0

        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        num_of_total_diffusion_calculated = 0  # without small diffusion and small A'

        print("Starting")

        # (path, graph_name, sep_char, P_range) = tup
        t = time.time()
        # G = read_network_from_file_simple(path, graph_name, sep_char)
        G = read_graph_from_file_Prange(path, graph_name, sep_char, P_range)
        tot = time.time() - t
        print("finished reading ", graph_name, "time:", tot)

        if graph_name in ["facebook_friendships", "facebook_nips", "twitter"]:  # added facebook_friendships
            min_size_of_diffusion = 10

        # output file:
        output_file = graph_name + ".txt"

        while num_of_total_diffusion_calculated < 1000:
            time_of_diffusion = time.time()
            G = reduce_graph_size(G, max_size_of_graph)

            print("The number of nodes in the reduced graph is: ", len(G.nodes), " and edges:", len(G.edges))

            # if len(G.nodes) < max_size_of_graph:
            #     if len(G.nodes) < (max_size_of_graph / 2):
            #         continue

            source_node = random.choice(list(G.nodes()))
            print("The source node is: ", source_node)
            print("Running the ic model")
            infected_nodes = simulate_ic_model(G, source_node, max_iterations=len(G.nodes))

            if len(infected_nodes) < min_size_of_diffusion:  # if the diffusion is smaller then 20
                print("Too small diffusion len(active set)= ", len(infected_nodes))
                print("The infected nodes are: ", infected_nodes)
                num_of_too_small_diffusion += 1
                continue  # check the next graph

            print(f"Number of the infected nodes is: {len(infected_nodes)} and number of edges is: "
                  f"{len(G.subgraph(infected_nodes).edges)}")
            infected_graph = create_induced_subgraph(G, infected_nodes)
            print("induced subgraph was created")
            possible_sources = Atag_calc(infected_graph)

            if len(possible_sources) <= 1:  # if A' is smaller then 2
                print("A Tag is smaller than 2")
                num_of_too_small_A_tag += 1
                continue  # check the next graph

            # print(f"Possible sources: {possible_sources}")
            print(f"Number of possible sources of infection: {len(possible_sources)}")
            print(f"source_node in possible_sources: {source_node in possible_sources}")
            induced_graph = create_induced_subgraph(G, possible_sources)
            print("Number of Edges is:", len(induced_graph.edges))

            print("graph induced")

            # going back to the start
            # reversed_G = reverse_and_normalize_weights(induced_graph)
            no_loops_G = reverse_and_normalize_weights(induced_graph)
            print("finished no loop method")
            # self_loops_G = roni_self_loops(induced_graph)
            Max_weight_arborescence_G = Max_weight_arborescence(induced_graph)
            print("graph reversed")

            if not verify_no_loops_transformation(induced_graph, no_loops_G):
                print(f"Did the no loops method work? false")
                # continue # the algorithm didn't work move to the next graph

            # if not verify_self_loops_transformation(induced_graph, self_loops_G):
            #     print(f"Did the self loops method work? false")
            #     # continue # the algorithm didn't work move to the next graph

            exact_match_results = evaluate_exact_match(source_node, no_loops_G,
                                                       Max_weight_arborescence_G, induced_graph)
            # naive_num_of_successes += exact_match_results[0]
            no_loop_num_of_successes += exact_match_results[0]  # if using naive and self loop change to 1
            # self_loop_num_of_successes += exact_match_results[2]
            max_arbo_num_of_successes += exact_match_results[1]  # if using naive and self loop change to 3

            print("finished the exact match method")

            top3_results = evaluate_top3(source_node, no_loops_G, Max_weight_arborescence_G)
            # naive_top3_successes += top3_results[0]
            no_loop_top3_successes += top3_results[0]  # if using naive and self loop change to 1
            # self_loop_top3_successes += top3_results[2]
            max_arbo_top3_successes += top3_results[1]  # if using naive and self loop change to 3

            print("finished the top 3 method")

            near_results = evaluate_near_source(source_node, no_loops_G, Max_weight_arborescence_G,
                                                induced_graph)
            # naive_near_successes += near_results[0]
            no_loop_near_successes += near_results[0]  # if using naive and self loop change to 1
            # self_loop_near_successes += near_results[2]
            max_arbo_near_successes += near_results[1]  # if using naive and self loop change to 1

            print(f"The real source is node {source_node}")

            num_of_total_diffusion_calculated += 1
            print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")
            tot = time.time() - time_of_diffusion
            print("time it took to calculate the diffusion is:", tot)

        total_time = time.time() - begin_time
        Append_to_file(output_file,
                       f"Results for graph {graph_name} (total good diffusions: {num_of_total_diffusion_calculated})\n")

        Append_to_file(output_file, "--- Evaluation 1: Exact match successes ---")
        # Append_to_file(output_file, f"Naive successes: {naive_num_of_successes}")
        Append_to_file(output_file, f"No loop successes: {no_loop_num_of_successes}")
        # Append_to_file(output_file, f"Self loop successes: {self_loop_num_of_successes}")
        Append_to_file(output_file, f"Max arborescence successes: {max_arbo_num_of_successes}\n")

        Append_to_file(output_file, "--- Evaluation 2: Source in top-3 successes ---")
        # Append_to_file(output_file, f"Naive top-3 successes: {naive_top3_successes}")
        Append_to_file(output_file, f"No loop top-3 successes: {no_loop_top3_successes}")
        # Append_to_file(output_file, f"Self loop top-3 successes: {self_loop_top3_successes}")
        Append_to_file(output_file, f"Max arborescence top-3 successes: {max_arbo_top3_successes}\n")

        Append_to_file(output_file, "--- Evaluation 3: Within 3-hop neighborhood successes ---")
        # Append_to_file(output_file, f"Naive near-source successes: {naive_near_successes}")
        Append_to_file(output_file, f"No loop near-source successes: {no_loop_near_successes}")
        # Append_to_file(output_file, f"Self loop near-source successes: {self_loop_near_successes}")
        Append_to_file(output_file, f"Max arborescence near-source successes: {max_arbo_near_successes}\n")

        Append_to_file(output_file, f"Number of too small diffusions: {num_of_too_small_diffusion}")
        Append_to_file(output_file, f"Number of too small A': {num_of_too_small_A_tag}")
        Append_to_file(output_file, f"Total time elapsed: {total_time:.2f} seconds")
        Append_to_file(output_file, "____________________________\n")


def evaluate_exact_match(source_node, no_loops_G, max_arbo_weights, induced_graph):
    # Find the most probable source for each method
    # print("starting algo naive")
    # naive_node, naive_max_prob = find_most_probable_source(reversed_G)
    # print("naive's prediction: ", naive_node)

    # print("starting no loops")
    no_loop_node, no_loop_max_prob = find_most_probable_source_no_loop(no_loops_G, induced_graph)
    print("no loops's prediction: ", no_loop_node)

    # print("starting self loops")
    # self_loop_node, self_loop_max_prob = find_most_probable_source(self_loops_G)
    # print("self loops's prediction: ", self_loop_node)

    # print("starting max arbo")
    max_arbo_node = max(max_arbo_weights, key=max_arbo_weights.get)
    print("max arbo's prediction: ", max_arbo_node)

    # naive_success = int(source_node == naive_node)
    no_loop_success = int(source_node == no_loop_node)
    # self_loop_success = int(source_node == self_loop_node)
    max_arbo_success = int(source_node == max_arbo_node)

    return no_loop_success, max_arbo_success


def evaluate_top3(source_node, no_loop_G, max_arbo_weights):
    # Get top 3 most probable sources for each method
    # naive_top3 = list(find_top_three(reversed_G))
    no_loop_top3 = list(find_top_three(no_loop_G))
    # self_loop_top3 = list(find_top_three(self_loops_G))
    max_arbo_top3 = sorted(max_arbo_weights, key=max_arbo_weights.get, reverse=True)[:3]

    # naive_top3_success = int(source_node in naive_top3)
    no_loop_top3_success = int(source_node in no_loop_top3)
    # self_loop_top3_success = int(source_node in self_loop_top3)
    max_arbo_top3_success = int(source_node in max_arbo_top3)

    return no_loop_top3_success, max_arbo_top3_success


def evaluate_near_source(source_node, no_loops_G, max_arbo_weights, induced_graph):
    # naive_near_success = int(is_most_probable_near_source(reversed_G, source_node))
    no_loop_near_success = int(is_most_probable_near_source_no_loop(no_loops_G, induced_graph, source_node))
    # self_loop_near_success = int(is_most_probable_near_source(self_loops_G, source_node))
    max_arbo_near_success = int(is_most_probable_near_source_max_arbo(max_arbo_weights, induced_graph, source_node))

    return no_loop_near_success, max_arbo_near_success


def reduce_graph_size(G: nx.DiGraph, maximum_size: int, max_edges: int = 300_000) -> nx.DiGraph:
    """
    Reduces the size of a directed graph while preserving connectivity and diffusion potential:
    1. Extracts the largest weakly connected component.
    2. Selects up to `maximum_size` nodes via BFS starting from a high out-degree node.
    3. If too many edges, randomly removes excess edges.
    """
    if G is None or maximum_size <= 0 or G.number_of_nodes() == 0:
        return nx.DiGraph()

    # Step 1: Get the largest weakly connected component
    if not nx.is_weakly_connected(G):
        largest_wcc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_wcc).copy()

    if G.number_of_nodes() <= maximum_size and G.number_of_edges() <= max_edges:
        return G.copy()

    # Step 2: BFS from high out-degree node
    sorted_nodes = sorted(G.nodes, key=lambda n: G.out_degree(n), reverse=True)
    attempts = 10000
    best_nodes = set()

    for _ in range(attempts):
        start_node = random.choice(sorted_nodes[:len(sorted_nodes) // 5])  # top 20% by out-degree
        visited = set()
        queue = deque([start_node])
        selected_nodes = set()

        while queue and len(selected_nodes) < maximum_size:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            selected_nodes.add(current)
            for neighbor in G.successors(current):
                if neighbor not in visited and neighbor not in queue and len(selected_nodes) < maximum_size:
                    queue.append(neighbor)

        if len(selected_nodes) > len(best_nodes):
            best_nodes = selected_nodes
            if len(best_nodes) == maximum_size:
                break  # can't improve further

    # Step 3: Create subgraph and prune edges
    subgraph = G.subgraph(best_nodes).copy()
    if subgraph.number_of_edges() > max_edges:
        edges = list(subgraph.edges)
        random.shuffle(edges)
        subgraph.remove_edges_from(edges[max_edges:])

    return subgraph




