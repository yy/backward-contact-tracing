import sys

import EoN
import networkx as nx
import numpy as np

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


if __name__ == "__main__":

    NET_SIZE = int(sys.argv[1])  # network size
    M = int(sys.argv[2])  # min degree
    trans_rate = float(sys.argv[3])  # transmission rate
    recov_rate = float(sys.argv[4])  # transmission rate
    output_log_file = sys.argv[5]
    output_net_file = sys.argv[6]

    G = nx.barabasi_albert_graph(NET_SIZE, M)

    params = {
        "tmax": 100,
        "tau": trans_rate,  # transmission rate (for edges)
        "gamma": recov_rate,  # recovery rate (for nodes)
        "rho": 0.001,  # initial seed fraction
    }
    sim_data = EoN.fast_SIR(G, return_full_data=True, **params)
    to_log(sim_data, output_log_file)
    nx.write_edgelist(G, output_net_file)
