import os
import sys

import networkx as nx
import numpy as np

import graph_tool.all as gt
import seir


def generate_network(N, gamma, min_deg):
    def deg_sampler(gamma, N, min_deg):
        """
        Generate Zipf-like random variables,
        but in inclusive [min...max] interval
        """
        v = np.arange(min_deg, N)  # values to sample
        p = 1.0 / np.power(v, gamma)  # probabilities
        p /= np.sum(p)  # normalized

        def sampler():
            d = np.random.choice(v, size=1, replace=True, p=p)
            return d

        return sampler

    dsampler = deg_sampler(gamma, N, min_deg)
    g = gt.random_graph(N, dsampler, directed=False, verbose=True)
    A = gt.adjacency(g).T
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph)
    return G


if __name__ == "__main__":

    N = int(sys.argv[1])
    gamma = float(sys.argv[2])
    E2I_rate = float(sys.argv[3])  # rate from E to I states
    trans_rate = float(sys.argv[4])  # transmission rate
    recov_rate = float(sys.argv[5])  # recovery rate
    output_log_file = sys.argv[6]
    output_net_file = sys.argv[7]
    min_deg = 2

    params = {
        "E2I_rate": E2I_rate,
        "trans_rate": trans_rate,  # transmission rate (for edges)
        "recov_rate": recov_rate,  # recovery rate (for nodes)
        "init_seed_frac": 0.001,  # initial seed fraction
    }

    if os.path.exists(output_net_file) is False:
        G = generate_network(N, gamma, min_deg)
    else:
        G = nx.read_edgelist(output_net_file, nodetype=int)
    sim_data = seir.run_SEIR(G, **params)
    seir.to_log(sim_data, output_log_file)
    nx.write_edgelist(G, output_net_file)
