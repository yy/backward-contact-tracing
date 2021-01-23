import json
import sys
from collections import Counter

import EoN
import networkx as nx
import numpy as np


def run_sir_contact_tracing(G, p_s, p_t, p_r, tmax, tau, gamma, rho, t_contact_tracing):
    """
    Parameters
    ----------
    G : networkx.Graph
        Base graph on which the SIR is simulated
    p_s : float
        sampling probability for infected individuals
    p_t : float
        probability for adding non-parent nodes.
        Parent nodes refer to the source node who infect the focal nodes
    p_r : float
        For each sampled infected node, we conduct contact tracing with probability p_r.
    tmax : int
        maximum simulation time
    tau  : fooat
        transimission rate
    gamma : float
        recovery rate
    rho : float
        initial seed fraction
    t_contact_tracing : float
        time at which we carry out the contact tracing
    """

    # Run a simulation
    sim_data = EoN.fast_SIR(
        G, return_full_data=True, tmax=tmax, tau=tau, gamma=gamma, rho=rho
    )
    T = sim_data.transmission_tree()

    # Find the infected indivisuals
    nodes_with_state_I = [
        node
        for node, state in sim_data.get_statuses(G.nodes(), t_contact_tracing).items()
        if state == "I"
    ]
    infected = [x for x in nodes_with_state_I if x in T and list(T.predecessors(x))]

    # Sample infected nodes
    infected_sampled = [x for x in infected if np.random.random() <= p_s]

    # Contact tracing
    parents = []
    to_be_traced = Counter()
    for node in infected_sampled:
        parent = list(T.predecessors(node))[0]
        parents.append(parent)
        temp = [
            x for x in G.neighbors(node) if x == parent or np.random.random() <= p_t
        ]
        for i in temp:
            if np.random.random() <= p_r:
                to_be_traced[i] += 1

    return {
        "infected": infected,
        "infected_sampled": infected_sampled,
        "parents": parents,
        "to_be_traced": dict(to_be_traced),
    }


if __name__ == "__main__":

    # Parse input
    BASE_NETWORK = sys.argv[1]  # base network file
    p_s = float(sys.argv[2])  # probability for sampling secondary cases
    p_t = float(sys.argv[3])  # probabiility for adding non-parent nodes
    p_r = float(sys.argv[4])  # tracing probability
    tmax = int(sys.argv[5])  # maximum simulation time
    tau = float(sys.argv[6])  # transmission rate
    gamma = float(sys.argv[7])  # recovery rate
    rho = float(sys.argv[8])  # initial seed fraction
    t_contact_tracing = int(sys.argv[9])  # time for contact tracing
    num_samples = int(sys.argv[10])  # number of samples
    resfilename = sys.argv[11]

    # Load the base network
    G = nx.read_edgelist(BASE_NETWORK)

    # Run simulations
    results = run_sir_contact_tracing(
        G, p_s, p_t, p_r, tmax, tau, gamma, rho, t_contact_tracing
    )

    # Save
    with open(resfilename, "w") as f:
        json.dump(results, f)
