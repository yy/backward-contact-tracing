import os
import sys

import EoN
import networkx as nx
import numpy as np

import graph_tool.all as gt
import pandas as pd


def to_log(sim_data, filename):
    df = pd.DataFrame(sim_data.transmissions(), columns=["elapsed", "source", "node"])
    df["type"] = "i"

    nodes = np.unique(df[["source", "node"]].values).astype(int)
    nodes = nodes[~np.isnan(nodes)]
    node_history = [(sim_data.node_history(node), node) for node in nodes]
    dg = pd.DataFrame(
        [
            {"elapsed": x[0][0][-1], "source": "", "node": x[1], "type": "r"}
            for x in node_history
            if "R" in x[0][1]
        ]
    )
    df = pd.concat([df.fillna(""), dg], ignore_index=True)
    df.to_csv(filename, sep=",", index=False)


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
    trans_rate = float(sys.argv[3])  # transmission rate
    recov_rate = float(sys.argv[4])  # recovery rate
    output_log_file = sys.argv[5]
    output_net_file = sys.argv[6]
    min_deg = 2

    params = {
        "tmax": 100,
        "tau": trans_rate,  # transmission rate (for edges)
        "gamma": recov_rate,  # recovery rate (for nodes)
        "rho": 0.001,  # initial seed fraction
    }

    if os.path.exists(output_net_file) is False:
        G = generate_network(N, gamma, min_deg)
    else:
        G = nx.read_edgelist(output_net_file, nodetype=int)
    sim_data = EoN.fast_SIR(G, return_full_data=True, **params)
    to_log(sim_data, output_log_file)
    nx.write_edgelist(G, output_net_file)
