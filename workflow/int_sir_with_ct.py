import json
import sys
from functools import partial

import networkx as nx

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
    infected_sampled, traced_contact = interv.contact_tracing(G, state, p_s, p_t, p_r)
    isolated = set(traced_contact.keys()) | set(infected_sampled)
    return isolated


if __name__ == "__main__":

    # Base network size
    NET_SIZE = int(sys.argv[1])
    C = int(sys.argv[2])  # parameter for the Price model
    M = int(sys.argv[3])  # parameter for the Price model

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

    def price_graph(N, c, m):
        g = gt.price_network(N, m=m, c=c, directed=False)
        A = gt.adjacency(g)
        return nx.from_scipy_sparse_matrix(A)

    # Evaluate the intervention
    (
        num_infected,
        num_isolated,
        num_new_cases,
        num_direct_prev_cases,
        num_indirect_prev_cases,
        num_prev_cases,
    ) = interv.eval_interv(
        partial(price_graph, N=NET_SIZE, m=M, c=C),
        rho,
        {"tau": tau, "gamma": gamma},
        t_interv,
        dt,
        partial(interv_by_ct, p_s=p_s, p_t=p_t, p_r=p_r),
        num_sim,
    )

    result = {
        "num_infected": list(num_infected),
        "num_isolated": list(num_isolated),
        "num_new_cases": list(num_new_cases),
        "num_direct_prev_cases": list(num_direct_prev_cases),
        "num_indirect_prev_cases": list(num_indirect_prev_cases),
        "num_prev_cases": list(num_prev_cases),
        "params": {
            "num_nodes": NET_SIZE,
            "price_c": C,
            "price_m": M,
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
