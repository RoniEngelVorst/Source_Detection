from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *
import time
import random

def Append_to_file(file_name, text):
    print(file_name, ":", text)
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(text + "\n")


def run_random_graphs():
    begin_time = time.time()

    random_graphs = [
        ("G1", 500, 0.1, 0.0416),
        ("G2", 1000, 0.1, 0.0204),
        ("G3", 2000, 0.1, 0.0101),
        ("G4", 3000, 0.1, 0.0071),
        ("G5", 4000, 0.1, 0.0052),
        ("G6", 5000, 0.1, 0.0041),
        ("G7", 500, 0.0416, 0.1),
        ("G8", 1000, 0.02, 0.1),
        ("G9", 2000, 0.0101, 0.1),
        ("G10", 3000, 0.0067, 0.1),
        ("G11", 4000, 0.0052, 0.1),
        ("G12", 5000, 0.0041, 0.1),
        ("G13", 10000, 0.002, 0.1),
        ("G14", 15000, 0.0013, 0.1)
    ]

    min_size_of_diffusion = 20
    max_diffusions_per_graph = 1000
    print("starting")

    for (graph_name, n_nodes, p_edge, p_weight) in random_graphs:
        G = random_graph_generator(n_nodes, p_edge, p_weight)

        # Initialize success counters
        naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0

        naive_top3_successes = 0
        no_loop_top3_successes = 0
        self_loop_top3_successes = 0
        max_arbo_top3_successes = 0

        naive_near_successes = 0
        no_loop_near_successes = 0
        self_loop_near_successes = 0
        max_arbo_near_successes = 0

        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        num_of_total_diffusion_calculated = 0

        output_file = graph_name + ".txt"

        while num_of_total_diffusion_calculated < max_diffusions_per_graph:
            source_node = random.choice(list(G.nodes()))
            infected_nodes = simulate_ic_model(G, source_node, max_iterations=len(G.nodes))

            if len(infected_nodes) < min_size_of_diffusion:
                print("too small diffusion")
                num_of_too_small_diffusion += 1
                continue

            infected_graph = create_induced_subgraph(G, infected_nodes)
            possible_sources = Atag_calc(infected_graph)

            if len(possible_sources) <= 1:
                print("too small A' ")
                num_of_too_small_A_tag += 1
                continue

            print("creating induced subgraph")
            induced_graph = create_induced_subgraph(G, possible_sources)

            reversed_G = reverse_and_normalize_weights(induced_graph)
            no_loops_G = reverse_and_normalize_weights(induced_graph)
            self_loops_G = apply_self_loop_method(induced_graph)
            max_arbo_weights = Max_weight_arborescence(induced_graph)

            # Run evaluations
            exact_match_results = evaluate_exact_match(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights, induced_graph)
            naive_num_of_successes += exact_match_results[0]
            no_loop_num_of_successes += exact_match_results[1]
            self_loop_num_of_successes += exact_match_results[2]
            max_arbo_num_of_successes += exact_match_results[3]

            print("finished the exact match method")

            top3_results = evaluate_top3(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights)
            naive_top3_successes += top3_results[0]
            no_loop_top3_successes += top3_results[1]
            self_loop_top3_successes += top3_results[2]
            max_arbo_top3_successes += top3_results[3]

            print("finished the top 3 method")

            near_results = evaluate_near_source(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights, induced_graph)
            naive_near_successes += near_results[0]
            no_loop_near_successes += near_results[1]
            self_loop_near_successes += near_results[2]
            max_arbo_near_successes += near_results[3]

            num_of_total_diffusion_calculated += 1
            print("finished the near source method")

        total_time = time.time() - begin_time

        Append_to_file(output_file, f"Results for graph {graph_name} (total good diffusions: {num_of_total_diffusion_calculated})\n")

        Append_to_file(output_file, "--- Evaluation 1: Exact match successes ---")
        Append_to_file(output_file, f"Naive successes: {naive_num_of_successes}")
        Append_to_file(output_file, f"No loop successes: {no_loop_num_of_successes}")
        Append_to_file(output_file, f"Self loop successes: {self_loop_num_of_successes}")
        Append_to_file(output_file, f"Max arborescence successes: {max_arbo_num_of_successes}\n")

        Append_to_file(output_file, "--- Evaluation 2: Source in top-3 successes ---")
        Append_to_file(output_file, f"Naive top-3 successes: {naive_top3_successes}")
        Append_to_file(output_file, f"No loop top-3 successes: {no_loop_top3_successes}")
        Append_to_file(output_file, f"Self loop top-3 successes: {self_loop_top3_successes}")
        Append_to_file(output_file, f"Max arborescence top-3 successes: {max_arbo_top3_successes}\n")

        Append_to_file(output_file, "--- Evaluation 3: Within 3-hop neighborhood successes ---")
        Append_to_file(output_file, f"Naive near-source successes: {naive_near_successes}")
        Append_to_file(output_file, f"No loop near-source successes: {no_loop_near_successes}")
        Append_to_file(output_file, f"Self loop near-source successes: {self_loop_near_successes}")
        Append_to_file(output_file, f"Max arborescence near-source successes: {max_arbo_near_successes}\n")

        Append_to_file(output_file, f"Number of too small diffusions: {num_of_too_small_diffusion}")
        Append_to_file(output_file, f"Number of too small A': {num_of_too_small_A_tag}")
        Append_to_file(output_file, f"Total time elapsed: {total_time:.2f} seconds")
        Append_to_file(output_file, "____________________________\n")


def evaluate_exact_match(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights, induced_graph):
    # Find the most probable source for each method
    naive_node, _ = find_most_probable_source(reversed_G)
    no_loop_node, _ = find_most_probable_source_no_loop(no_loops_G, induced_graph)
    self_loop_node, _ = find_most_probable_source(self_loops_G)
    max_arbo_node = max(max_arbo_weights, key=max_arbo_weights.get)

    naive_success = int(source_node == naive_node)
    no_loop_success = int(source_node == no_loop_node)
    self_loop_success = int(source_node == self_loop_node)
    max_arbo_success = int(source_node == max_arbo_node)

    return naive_success, no_loop_success, self_loop_success, max_arbo_success


def evaluate_top3(source_node, reversed_G, no_loop_G, self_loops_G, max_arbo_weights):
    # Get top 3 most probable sources for each method
    naive_top3 = list(find_top_three(reversed_G))
    no_loop_top3 = list(find_top_three(no_loop_G))
    self_loop_top3 = list(find_top_three(self_loops_G))
    max_arbo_top3 = sorted(max_arbo_weights, key=max_arbo_weights.get, reverse=True)[:3]

    naive_top3_success = int(source_node in naive_top3)
    no_loop_top3_success = int(source_node in no_loop_top3)
    self_loop_top3_success = int(source_node in self_loop_top3)
    max_arbo_top3_success = int(source_node in max_arbo_top3)

    return naive_top3_success, no_loop_top3_success, self_loop_top3_success, max_arbo_top3_success


def evaluate_near_source(source_node, reversed_G, no_loops_G, self_loops_G, max_arbo_weights, induced_graph):
    naive_near_success = int(is_most_probable_near_source(reversed_G, source_node))
    no_loop_near_success = int(is_most_probable_near_source_no_loop(no_loops_G, induced_graph, source_node))
    self_loop_near_success = int(is_most_probable_near_source(self_loops_G, source_node))
    max_arbo_near_success = int(is_most_probable_near_source_max_arbo(max_arbo_weights, induced_graph, source_node))

    return naive_near_success, no_loop_near_success, self_loop_near_success, max_arbo_near_success


