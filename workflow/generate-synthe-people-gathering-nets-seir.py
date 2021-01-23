import os
import sys

import networkx as nx
import numpy as np

import graph_tool.all as gt
import seir


def generate_power_law_bipartite_net(
    N, frac_left_node, gamma, ave_deg, min_deg_left, min_deg_right, node_class
):
    """
    Generate power law bipartite network.

    Params
    ------
    N : int
        Number of nodes
    frac_left_node : float
        Fraction of nodes on the left part.
    gamma : float
        Power-law exponent (same expoent for both sides)
    ave_deg : float
        Average degree
    min_deg_left : int
        Minimum degree for nodes on the left part
    min_deg_right : int
        Minimum degree for nodes on the right part
    node_class : list of str
        Name of the class for the left and right nodes
        node_class[0] : str for left nodes.
        node_class[1] : str for right nodes.

    Return
    ------
    G : networkx.Graph
    """

    def zipf(a, min, max, size=None):
        """
        Generate Zipf-like random variables,
        but in inclusive [min...max] interval
        """
        v = np.arange(min, max + 1)  # values to sample
        p = 1.0 / np.power(v, a)  # probabilities
        p /= np.sum(p)  # normalized
        return np.random.choice(v, size=size, replace=True, p=p)

    def add_n_stabs(deg, n):
        """
        Add n stabs to degree sequence
        """
        stubs = np.random.choice(
            np.arange(len(deg)), size=int(n), replace=True, p=deg / np.sum(deg)
        )
        for s in stubs:
            deg[s] += 1
        return deg

    def to_graphical_deg_seq(deg_left, deg_right):
        """
        Make the degree sequence to be graphical
        by adding edges
        """
        deg_left_sum = np.sum(deg_left)
        deg_right_sum = np.sum(deg_right)

        if deg_left_sum < deg_right_sum:
            deg_left = add_n_stabs(deg_left, deg_right_sum - deg_left_sum)
        elif deg_left_sum > deg_right_sum:
            deg_right = add_n_stabs(deg_right, deg_left_sum - deg_right_sum)

        return deg_left, deg_right

    # Compute the number of nodes
    N_left = int(N * frac_left_node)
    N_right = N - N_left

    # Generate degree sequence
    deg_left = zipf(3, min_deg_left, N_right, size=N_left)
    deg_right = zipf(3, min_deg_right, N_left, size=N_right)

    # Rescale such that the average degree is the prescribed average degree
    E = ave_deg * (N_left + N_right)
    deg_left = np.clip(np.round(deg_left * E / np.sum(deg_left)), min_deg_left, N_right)
    deg_right = np.clip(
        np.round(deg_right * E / np.sum(deg_right)), min_deg_right, N_left
    )

    # Make them graphical degree sequences
    deg_left, deg_right = to_graphical_deg_seq(deg_left, deg_right)

    # Prepare parameters for graph-tool
    E = np.sum(deg_right)
    gt_params = {
        "out_degs": np.concatenate([np.zeros_like(deg_left), deg_right]).astype(int),
        "in_degs": np.concatenate([deg_left, np.zeros_like(deg_right)]).astype(int),
        "b": np.concatenate([np.zeros(N_left), np.ones(N_right)]),
        "probs": np.array([[0, 0], [E, 0]]),
        "directed": True,
        "micro_degs": True,
    }

    # Generate the network until the degree sequence
    # satisfied the thresholds
    while True:
        g = gt.generate_sbm(**gt_params)
        A = gt.adjacency(g).T
        A.data = np.ones_like(A.data)
        outdeg = np.array(A.sum(axis=1)).reshape(-1)[N_left:]
        indeg = np.array(A.sum(axis=0)).reshape(-1)[:N_left]
        if (np.min(indeg) >= min_deg_left) and (np.min(outdeg) >= min_deg_right):
            break

    # Convert to the networkx objet
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.Graph)

    # Add attributes to the nodes
    node_attr = {i: node_class[int(i > N_left)] for i in range(N)}
    nx.set_node_attributes(G, node_attr, "class")
    return G


if __name__ == "__main__":

    N = int(sys.argv[1])
    gamma = float(sys.argv[2])
    frac_gather = float(sys.argv[3])
    E2I_rate = float(sys.argv[4])  # rate from E to I states
    trans_rate = float(sys.argv[5])  # transmission rate
    recov_rate = float(sys.argv[6])  # recovery rate
    output_log_file = sys.argv[7]
    output_net_file = sys.argv[8]

    min_deg_left = 2
    min_deg_right = 1
    ave_deg = 5

    params = {
        "E2I_rate": E2I_rate,
        "trans_rate": trans_rate,  # transmission rate (for edges)
        "recov_rate": recov_rate,  # recovery rate (for nodes)
        "init_seed_frac": 0.001,  # initial seed fraction
    }

    if os.path.exists(output_net_file) is False:
        G = generate_power_law_bipartite_net(
            N,
            frac_gather,
            gamma,
            ave_deg,
            min_deg_left,
            min_deg_right,
            ["gathering", "people"],
        )
    else:
        G = nx.read_edgelist(edgelist, nodetype=int)

    sim_data = seir.run_SEIR(G, **params)
    seir.to_log(sim_data, output_log_file)
    nx.write_gexf(G, output_net_file)
