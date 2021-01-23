import json
import sys
from functools import partial

import networkx as nx
import numpy as np

import graph_tool.all as gt
import int_sir_with_interv as interv


def interv_by_ct(G, state, p_s, p_t, p_r):
    """
    Intervention by contact tracing

    Parameters
    ----------
    G : nx.networkx
        Base network
    state : State
        State of SIR model. See sir_with_interv module
    p_s : float
      Sampling probability for the infected individuals
    p_t : float
      Sampling probability for appending non-parent neighbors
    p_r : float
      Probability that the contact tracing is conducted

    Returns
    -------
    isolated : list
        List of nodes to be isolated by the intervention
    """
    infected_sampled, traced_contact = interv.contact_tracing(
        G, state, p_s, p_t, p_r, {"class": "people"}
    )
    isolated = set(traced_contact.keys()) | set(infected_sampled)
    return isolated


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

    # Parameter for the barabashi albert model
    NET_SIZE = int(sys.argv[1])  # network
    m = int(sys.argv[2])  # number of edges
    frac_gathr = float(sys.argv[3])  # fraction of gathering nodes

    # Parameters for the SIR model
    rho = float(sys.argv[4])  # frac of initial infected individuas
    tau = float(sys.argv[5])  # Transmission rate
    gamma = float(sys.argv[6])  # Recovery rate

    # Parameters for the intervention
    t_interv = float(sys.argv[7])  # time at which intervention is made
    dt = float(
        sys.argv[8]
    )  # time lag after which we measure the efficacy of the intervention
    p_s = float(sys.argv[9])  # sampling probability for the infected individuals
    p_t = float(
        sys.argv[10]
    )  # Probabbility of adding non-parent neighbors to the contact list
    p_r = float(sys.argv[11])  # Probability that the contact tracing is conducted

    # Parameters for the simulation
    num_sim = int(sys.argv[12])  # Number of simulation

    OUTPUT_FILE = sys.argv[13]

    # Evaluate the intervention
    (
        num_infected,
        num_isolated,
        num_new_cases,
        num_direct_prev_cases,
        num_indirect_prev_cases,
        num_prev_cases,
        num_nodes,
    ) = interv.eval_interv(
        partial(
            generate_power_law_bipartite_net,
            N=NET_SIZE,
            frac_left_node=frac_gathr,
            gamma=3,
            ave_deg=m,
            min_deg_left=2,
            min_deg_right=1,
            node_class=["gathering", "people"],
        ),
        rho,
        {"tau": tau, "gamma": gamma},
        t_interv,
        dt,
        partial(interv_by_ct, p_s=p_s, p_t=p_t, p_r=p_r),
        num_sim,
        {"class": "people"},
    )

    result = {
        "num_infected": list(num_infected),
        "num_isolated": list(num_isolated),
        "num_new_cases": list(num_new_cases),
        "num_direct_prev_cases": list(num_direct_prev_cases),
        "num_indirect_prev_cases": list(num_indirect_prev_cases),
        "num_prev_cases": list(num_prev_cases),
        "params": {
            "num_nodes": list(num_nodes),
            "price_m": m,
            "frac_gathr": frac_gathr,
            "rho": rho,
            "tau": tau,
            "gamma": gamma,
            "t_interv": t_interv,
            "dt": dt,
            "p_s": p_s,
            "p_t": p_t,
            "p_r": p_r,
            "num_sim": num_sim,
        },
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f)
