from Graph_Generator import *
from Independent_Cascade import *
from Marcov_Chains import *
from Real_Graphs import *
from Random_Graphs import *



def main():

    begin_time = time.time()
    # run_real_graphs()
    run_random_graphs()

    total_time = time.time() - begin_time



if __name__ == '__main__':
    main()