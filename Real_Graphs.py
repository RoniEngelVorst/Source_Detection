import networkx as nx
import numpy as np
import time
#my moduls:
from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *
import random

def Append_to_file(file_name, text):
    print(file_name, ":", text)
    file = open(file_name,"a")
    file.write(text + "\n")
    file.close()


def run_real_graph():
    begin_time = time.time()

    real_graphs = [
        # (r"real_graphs\youtube-links\out.youtube-links" , "youtube_links" , " " , 2) ,  # need to run on server
        # (r"real_graphs\youtube_friendship\out.com-youtube", "youtube_friendships", " ", 4),  # Not in the table
        # (r"real_graphs\ego-twitter\out.ego-twitter", "twitter", "\t", 100), # need to run on server
        # (r"real_graphs\slashdot\out.matrix", "slashdot", " ", 3),  # What Yeal wrote. but it is not in the folder
        # (r"real_graphs\slashdot-threads\out.slashdot-threads", "slashdot", " ", 3),  # Our fix. need to check
        # (r"real_graphs\ego-gplus\out.ego-gplus", "google_plus", "\t", 7), # need to run on server
        # (r"real_graphs\facebook_nips\out.ego-facebook", "facebook_nips", " ", 10), Not in the article
        # (r"real_graphs\facebook-wosn-links\out.facebook-wosn-links", "facebook_friendships", " ", 4), # need to run on server
         (r"real_graphs\soc-Epinions1\out.soc-Epinions1", "epinion_trust", "\t", 3),  # need to run on server
        # (r"real_graphs\digg-friends\out.digg-friends", "digg", " ", 3),  # need to run on server
        # (r"real_graphs\advogato\out.advogato", "advogato", " ", 4),  # need to run on server
    ]


    for tup in real_graphs:
        naive_num_of_successes = 0
        no_loop_num_of_successes = 0
        self_loop_num_of_successes = 0
        max_arbo_num_of_successes = 0
        num_of_too_small_diffusion = 0
        num_of_too_small_A_tag = 0
        min_size_of_diffusion = 20
        num_of_total_diffusion_calculated = 0  # without small diffusion and small A'
        print("Starting")

        (path, graph_name, sep_char, steps) = tup
        t = time.time()
        G = read_network_from_file_simple(path, graph_name, sep_char)
        tot = time.time() - t
        print("finished reading ", graph_name, "time:", tot)

        if graph_name in ["facebook_nips", "twitter"]:
            min_size_of_diffusion = 10

        # output file:
        output_file = tup[0] + ".txt"

        while num_of_total_diffusion_calculated < 1000:
            source_node = random.choice(list(G.nodes()))
            print("The source node is: ", source_node)
            print("Running the ic model")
            infected_nodes = simulate_ic_model(G, source_node, max_iterations=len(G.nodes))

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

            naive_most_probable_node, naive_max_prob = find_most_probable_source(reversed_G)
            no_loop_most_probable_node, no_loop_max_prob = find_most_probable_source_no_loop(no_loops_G, induced_graph)
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

            num_of_total_diffusion_calculated += 1
            print(f"The number of total diffusion calculated is: {num_of_total_diffusion_calculated} ")

        total_time = time.time() - begin_time
        Append_to_file(file_name=output_file,
                       text=f"The number of Successes in naive is: {naive_num_of_successes} ")
        Append_to_file(file_name=output_file,
                       text=f"The number of Successes in no loop is: {no_loop_num_of_successes} ")
        Append_to_file(file_name=output_file,
                       text=f"The number of Successes in self loop is: {self_loop_num_of_successes}")
        Append_to_file(file_name=output_file,
                       text=f"The number of Max weight arborescence is: {max_arbo_num_of_successes}")
        Append_to_file(file_name=output_file,
                       text="Number of too small diffusion is: " + str(num_of_too_small_diffusion))
        Append_to_file(file_name=output_file,
                       text="Number of too small A' is: " + str(num_of_too_small_A_tag))
        Append_to_file(file_name=output_file,
                       text="The total time is: " + str(total_time))

        Append_to_file(file_name=output_file,
                       text="number of to small diffusions: " + str(num_of_too_small_diffusion))
        Append_to_file(file_name=output_file,
                       text="number of times where |A'| was 1 (so all the algorithms are basically the same):" +
                            str(num_of_too_small_A_tag))
        Append_to_file(file_name=output_file,
                       text="number of 'good' diffusions:" + str(num_of_total_diffusion_calculated))
        Append_to_file(file_name=output_file, text="____________________________\n")



