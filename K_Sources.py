from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *

# ****** Method 1 - Top K-Sources ******

def Run_Top_K_Sources_On_Random_Graphs(k):
    begin_time = time.time()
    G1 = random_graph_generator(500, 0.1, 0.0416)
    # G2 = random_graph_generator(1000, 0.1, 0.0204)
    # G3 = random_graph_generator(1000, 0.1, 0.0204)
    # G4 = random_graph_generator(3000, 0.1, 0.0071)
    # G5 = random_graph_generator(4000, 0.1, 0.0052)
    # G6 = random_graph_generator(5000, 0.1, 0.0041)
    # G7 = random_graph_generator(500, 0.0416, 0.1)
    # G8 = random_graph_generator(1000, 0.02, 0.1)
    # G9 = random_graph_generator(2000, 0.0101, 0.1)
    # G10 = random_graph_generator(3000, 0.0067, 0.1)
    # G11 = random_graph_generator(4000, 0.0052, 0.1)
    # G12 = random_graph_generator(5000, 0.0041, 0.1)
    # G13 = random_graph_generator(10000, 0.002, 0.1)
    # G14 = random_graph_generator(5000, 0.0013, 0.1)

    random_graphs = [
        ("G1", 500, 0.1, 0.0416),
        # ("G2", 1000, 0.1, 0.0204),
        # ("G3", 2000, 0.1, 0.0101),
        # ("G4", 3000, 0.1, 0.0071),
        # ("G5", 4000, 0.1, 0.0052),
        # ("G6", 5000, 0.1, 0.0041),
        # ("G7", 500, 0.0416, 0.1),
        # ("G8", 1000, 0.02, 0.1),
        # ("G9", 2000, 0.0101, 0.1),
        # ("G10", 3000, 0.0067, 0.1),
        # ("G11", 4000, 0.0052, 0.1),
        # ("G12", 5000, 0.0041, 0.1),
        # ("G13", 10000, 0.002, 0.1),
        # ("G14", 15000, 0.0013, 0.1)
    ]
    Find_Top_K_Sources_Random(random_graphs, k)
    total_time = time.time() - begin_time
    print("The total time is: " + str(total_time))


def Find_Top_K_Sources_Random(graphs, k):
    num_of_total_diffusion_calculated = 0  # without small diffusion and small A'
    print("Starting")
    for tuple1 in graphs:
        G = random_graph_generator(tuple1[1], tuple1[2], tuple1[3])

        naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0
        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        min_size_of_diffusion = 20

        # # output file:
        # output_file = tuple1[0] + ".txt"

        while num_of_total_diffusion_calculated < 1000:
            source_nodes = random.sample(list(G.nodes()), k)
            print("The source nodes are: ", source_nodes)
            print("Running the ic model")
            infected_nodes = simulate_ic_model_on_k_sources(G, source_nodes, max_iterations=len(G.nodes))

            if len(infected_nodes) < min_size_of_diffusion:  # if the diffusion is bigger then 20
                print("Too small diffusion len(active set)= ", len(infected_nodes))
                print("The infected nodes are: ", infected_nodes)
                num_of_too_small_diffusion += 1
                continue  # check the next graph

            print(f"Number of the infected nodes is: {len(infected_nodes)}")
            infected_graph = create_induced_subgraph(G, infected_nodes)
            possible_sources = Atag_calc(infected_graph)

            if len(possible_sources) <= 1:  # if A' is smaller then 2
                num_of_too_small_A_tag += 1
                continue  # check the next graph

            print(f"Possible sources: {possible_sources}")
            print(f"Number of possible sources of infection: {len(possible_sources)}")
            induced_graph = create_induced_subgraph(G, possible_sources)

            reversed_G = reverse_and_normalize_weights(induced_graph)
            no_loops_G = reverse_and_normalize_weights(induced_graph)
            self_loops_G = apply_self_loop_method(induced_graph)
            Max_weight_arborescence_G = Max_weight_arborescence(induced_graph)

            if not verify_no_loops_transformation(induced_graph, no_loops_G):
                print(f"Did the no loops method work? false")
            #     continue # the algorithm didn't work move to the next graph
            #
            if not verify_self_loops_transformation(induced_graph, self_loops_G):
                print(f"Did the self loops method work? false")
            #     continue # the algorithm didn't work move to the next graph

            naive_most_probable_nodes, naive_max_probs = find_K_most_probable_sources(reversed_G, k)
            print(f"naive probable sources  {naive_most_probable_nodes}")
            no_loop_most_probable_nodes, no_loop_max_probs = find_K_most_probable_sources_no_loop(no_loops_G, induced_graph, k)
            print(f"No loops probable sources {no_loop_most_probable_nodes}")
            self_loop_most_probable_nodes, self_loop_max_probs = find_K_most_probable_sources(self_loops_G, k)
            print(f"Self loops probable sources {self_loop_most_probable_nodes}")
            top_k_arbo_nodes = sorted(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get, reverse=True)[:k]
            print(f"Arbo probable sources {top_k_arbo_nodes}")
            top_k_arbo_probs = [Max_weight_arborescence_G[node] for node in top_k_arbo_nodes]

            print(f"The real sources are nodes {source_nodes}")

            if all(node in naive_most_probable_nodes for node in source_nodes):
                naive_num_of_successes += 1

            if all(node in no_loop_most_probable_nodes for node in source_nodes):
                no_loop_num_of_successes += 1

            if all(node in self_loop_most_probable_nodes for node in source_nodes):
                self_loop_num_of_successes += 1

            if all(node in top_k_arbo_nodes for node in source_nodes):
                max_arbo_num_of_successes += 1

            num_of_total_diffusion_calculated += 1
            print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")

        print(f"The number of Successes in naive is: {naive_num_of_successes} ")
        print(f"The number of Successes in no loop is: {no_loop_num_of_successes} ")
        print(f"The number of Successes in self loop is: {self_loop_num_of_successes} ")
        print(f"The number of Max weight arborescence is: {max_arbo_num_of_successes}")
        print("number of 'good' diffusions:" + str(num_of_total_diffusion_calculated))