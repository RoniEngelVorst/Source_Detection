from Graph_Generator import *


def main():
    # Generate the graph
    G = random_graph_generator(1000, 0.1, 0.0416)

    # Visualize the graph
    visualize_subgraph(G,30)






if __name__ == '__main__':
    main()