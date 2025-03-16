from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *


def main():

    begin_time = time.time()

    G1 = random_graph_generator(500,0.1,0.0416)
    G2 = random_graph_generator(1000, 0.1, 0.0204)
    G3 = random_graph_generator(1000, 0.1, 0.0204)
    G4 = random_graph_generator(3000, 0.1, 0.0071)
    G5 = random_graph_generator(4000, 0.1, 0.0052)
    G6 = random_graph_generator(5000, 0.1, 0.0041)
    G7 = random_graph_generator( 500, 0.0416, 0.1)
    G8 = random_graph_generator(1000, 0.02, 0.1)
    G9 = random_graph_generator(2000, 0.0101, 0.1)
    G10 = random_graph_generator(3000, 0.0067, 0.1)
    G11 = random_graph_generator(4000, 0.0052, 0.1)
    G12 = random_graph_generator(5000, 0.0041, 0.1)
    G13 = random_graph_generator(10000, 0.002, 0.1)
    G14 = random_graph_generator(5000, 0.0013, 0.1)

    naive_num_of_successes = 0
    no_loop_num_of_successes = 0
    self_loop_num_of_successes = 0
    max_arbo_num_of_successes = 0
    num_of_too_small_diffusion = 0
    num_of_too_small_A_tag = 0
    min_size_of_diffusion = 20
    num_of_total_diffusion_calculated = 0 # without small diffusion and small A'
    print("Starting")
    while num_of_total_diffusion_calculated < 1000:
        source_node = random.choice(list(G1.nodes()))
        print("The source node is: ", source_node)
        print("Running the ic model")
        infected_nodes = simulate_ic_model(G1, source_node, max_iterations=len(G1.nodes))

        if len(infected_nodes) < min_size_of_diffusion: # if the diffusion is bigger then 20
            print("Too small diffusion len(active set)= ", len(infected_nodes))
            print("The infected nodes are: ", infected_nodes)
            num_of_too_small_diffusion += 1
            continue # check the next graph

        print(f"Number of the infected nodes is: {len(infected_nodes)}")
        infected_graph = create_induced_subgraph(G1, infected_nodes)
        possible_sources = Atag_calc(infected_graph)

        if len(possible_sources) <= 1: # if A' is smaller then 2
            num_of_too_small_A_tag += 1
            continue # check the next graph

        print(f"Possible sources: {possible_sources}")
        print(f"Number of possible sources of infection: {len(possible_sources)}")
        induced_graph = create_induced_subgraph(G1, possible_sources)

        reversed_G = reverse_and_normalize_weights(induced_graph)
        no_loops_G = apply_no_loops_method(induced_graph)
        self_loops_G = apply_self_loop_method(induced_graph)
        Max_weight_arborescence_G = Max_weight_arborescence(induced_graph)

        if not verify_no_loops_transformation(induced_graph, no_loops_G):
            print(f"Did the no loops method work? false")
        #     continue # the algorithm didn't work move to the next graph
        #
        if not verify_self_loops_transformation(induced_graph, self_loops_G):
            print(f"Did the self loops method work? false")
        #     continue # the algorithm didn't work move to the next graph

        naive_most_probable_node, naive_max_prob = find_most_probable_source(reversed_G)
        no_loop_most_probable_node, no_loop_max_prob = find_most_probable_source_no_loop(no_loops_G, G1)
        self_loop_most_probable_node, self_loop_max_prob = find_most_probable_source(self_loops_G)
        max_arbo_most_probable_node = max(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get)
        max_arbo_max_prob = Max_weight_arborescence_G[max_arbo_most_probable_node]

        print(f"The real source is node {source_node}")

        if source_node is naive_most_probable_node:
            naive_num_of_successes += 1

        if source_node is no_loop_most_probable_node:
            no_loop_num_of_successes += 1

        if source_node is self_loop_most_probable_node:
            self_loop_num_of_successes += 1

        if source_node is max_arbo_most_probable_node:
            max_arbo_num_of_successes += 1

        # ********************************** the second way of comparing with the top 3 ******************************

        # naive_most_probable_node, naive_max_prob = find_top_three(reversed_G)
        # # no_loop_most_probable_node, no_loop_max_prob = find_top_three(no_loops_G, G1)
        # self_loop_most_probable_node, self_loop_max_prob = find_top_three(self_loops_G)
        # max_arbo_most_probable_node = max(Max_weight_arborescence_G, key=Max_weight_arborescence_G.get)
        # max_arbo_max_prob = Max_weight_arborescence_G[max_arbo_most_probable_node]
        #
        # print(f"The real source is node {source_node}")
        #
        # if any(source_node == node for node, _ in naive_most_probable_node):
        #     naive_num_of_successes += 1
        #
        # # if any(source_node == node for node, _ in no_loop_most_probable_node):
        # #     no_loop_num_of_successes += 1
        #
        # if any(source_node == node for node, _ in self_loop_most_probable_node):
        #     self_loop_num_of_successes += 1
        #
        # if source_node is max_arbo_most_probable_node:
        #     max_arbo_num_of_successes += 1

        # *************************** the third comparing with the nearest up to 3 *************************************

        # naive_most_probable_node = is_most_probable_near_source(reversed_G, source_node)
        # no_loop_most_probable_node = is_most_probable_near_source_no_loop(no_loops_G, G1, source_node)
        # self_loop_most_probable_node = is_most_probable_near_source(self_loops_G, source_node)
        # max_arbo_most_probable_node = is_most_probable_near_source_max_arbo(Max_weight_arborescence_G)
        #
        # print(f"The real source is node {source_node}")
        #
        # if naive_most_probable_node:
        #     naive_num_of_successes += 1
        #
        # if no_loop_most_probable_node:
        #     no_loop_num_of_successes += 1
        #
        # if self_loop_most_probable_node:
        #     self_loop_num_of_successes += 1
        #
        # if max_arbo_most_probable_node:
        #     max_arbo_num_of_successes += 1

        # ********************************************** end **********************************************************

        num_of_total_diffusion_calculated += 1
        print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")

    total_time = time.time() - begin_time
    print(f"The number of Successes in naive is: {naive_num_of_successes} ")
    print(f"The number of Successes in no loop is: {no_loop_num_of_successes} ")
    print(f"The number of Successes in self loop is: {self_loop_num_of_successes}")
    print(f"The number of Max weight arborescence is: {max_arbo_num_of_successes}")
    print("Number of too small diffusion is: ", num_of_too_small_diffusion)
    print("Number of too small A' is: ", num_of_too_small_A_tag)
    print("The total time is: ", total_time)


if __name__ == '__main__':
    main()