import networkx as nx
import sys

if __name__ == "__main__":
    RES_FILENAME = sys.argv[1]
    G = nx.barabasi_albert_graph(10 ** 6, 5)
    nx.write_edgelist(G, RES_FILENAME)
