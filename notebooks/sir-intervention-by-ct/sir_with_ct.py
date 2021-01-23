import utils_sir_with_ct as utils
from utils_sir_with_ct import State
import numpy as np
import networkx as nx
from functools import partial
import sys
import json


def interv_by_ct(G, state, p_s, p_t, p_r):
    """
    Intervention by contact tracing 

    Parameters
    ----------
    G : nx.networkx
        Base network 
    state : State 
        State of SIR model. See utils_sir_with_ct module
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
    infected_sampled, traced_contact = utils.contact_tracing(G, state, p_s, p_t, p_r)
    isolated = set(traced_contact.keys()) | set(infected_sampled)
    return isolated


if __name__ == "__main__":

    # Base network
    BASE_NET_FILE = sys.argv[1]

    # Parameters for the SIR model
    rho = float(sys.argv[2])  # frac of initial infected individuas
    tau = float(sys.argv[3])  # Transmission rate
    gamma = float(sys.argv[4])  # Recovery rate

    # Parameters for the intervention
    t_interv = float(sys.argv[5])  # time at which intervention is made
    dt = float(
        sys.argv[6]
    )  # time lag after which we measure the efficacy of the intervention
    p_s = float(sys.argv[7])  # sampling probability for the infected individuals
    p_t = float(
        sys.argv[8]
    )  # Probabbility of adding non-parent neighbors to the contact list
    p_r = float(sys.argv[9])  # Probability that the contact tracing is conducted

    # Parameters for the simulation
    num_sim = int(sys.argv[10])  # Number of simulation

    OUTPUT_FILE = sys.argv[11]

    G = nx.read_edgelist(BASE_NET_FILE)

    # Evaluate the intervention
    (
        num_infected,
        num_isolated,
        num_new_cases,
        num_direct_prev_cases,
        num_indirect_prev_cases,
        num_prev_cases,
    ) = utils.eval_interv(
        G,
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
            "BASE_NET_FILE": BASE_NET_FILE,
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
