from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *
from Real_Graphs import *
from Random_Graphs import *
from K_Sources import *
from K_Sources_Random import *




def main():

    begin_time = time.time()
    # run_real_graphs()
    # run_random_graphs()
    # Run_Top_K_Sources_On_Random_Graphs(k=1)
    run_k_sources_all_methods_on_Random()



    total_time = time.time() - begin_time



if __name__ == '__main__':
    main()