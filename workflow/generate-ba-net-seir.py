import sys

import networkx as nx

import seir

if __name__ == "__main__":

    NET_SIZE = int(sys.argv[1])  # network size
    M = int(sys.argv[2])  # min degree
    E2I_rate = float(sys.argv[3])  # rate from E to I states
    trans_rate = float(sys.argv[4])  # transmission rate
    recov_rate = float(sys.argv[5])  # recovery rate
    output_log_file = sys.argv[6]
    output_net_file = sys.argv[7]

    G = nx.barabasi_albert_graph(NET_SIZE, M)

    params = {
        "E2I_rate": E2I_rate,
        "trans_rate": trans_rate,  # transmission rate (for edges)
        "recov_rate": recov_rate,  # recovery rate (for nodes)
        "init_seed_frac": 0.001,  # initial seed fraction
    }
    sim_data = seir.run_SEIR(G, **params)
    seir.to_log(sim_data, output_log_file)
    nx.write_edgelist(G, output_net_file)
