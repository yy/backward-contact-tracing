"""
Utility functions for SIR simulations with intervention by contact tracing.

Notes
-----
- We assume the SIR model for exponentially distributed infection and recovery times.
- The utility functions process a `state` variable, which is represented by
    a python dict object (see `init_state` function)
- The `state` variable contains all the information needed to run the SIR simulations,
    contact tracing and isolations by the contact tracing.
"""

import copy
from collections import Counter
from functools import reduce
from operator import or_

import EoN
import networkx as nx
import numpy as np


class State:
    def __init__(self):
        self.infected = []  # list of infected nodes
        self.recovered = []  # list of recovered nodes
        self.isolated = []  # list of isolated nodes by intervention
        self.trans_tree = None  # transmission tree
        self.t = 0  # time counter

    def init_state(self, G, rho, init_node_attr=None):
        """
        Initialize the state object

        Parameters
        ----------
        G : networkx.Graph
            Base graph
        rho : float
            Fraction of initial infected individuals
        init_node_attr: dict
            todo add desc

        Returns
        -------
        state : State
        """
        if init_node_attr is None:
            nodes = list(G.nodes())
        else:
            k = list(init_node_attr.keys())[0]
            v = list(init_node_attr.values())[0]
            nodes = [x for x in G.nodes() if G.nodes[x][k] == v]

        self.infected = np.random.choice(
            nodes, np.random.binomial(len(nodes), rho), replace=False
        )
        return self

    def isolate_nodes(self, node_set):
        """
        Isolate nodes from the simulation

        Parameters
        ----------
        node_set : list
            List of nodes to which isolations apply

        Returns
        -------
        state : State
            the state object after isolations
        """
        if type(node_set) != "list":
            node_set = list(node_set)
        self.isolated = list(set(self.isolated + node_set))
        return self


def run_sir(G, dt, sir_params, state):
    """
    Run simulation for time dt

    Parameters
    ----------
    sim_params : dict
        Parameter for the fast_SIR
    state : State
        state of SIR simulation

    Returns
    -------
    sim_data : simulation data
    state : dict
        the state object after running simulation for dt
    """
    sir_params["tmax"] = dt
    sim_data = EoN.fast_SIR(
        G,
        return_full_data=True,
        initial_infecteds=state.infected,
        initial_recovereds=state.recovered + state.isolated,
        **sir_params
    )

    # Update node state
    node_state = sim_data.get_statuses(G.nodes(), dt)
    infected = [k for k, v in node_state.items() if v == "I"]
    recovered = [k for k, v in node_state.items() if v == "R"]
    state.infected = infected
    state.recovered = list(set(recovered) - set(state.isolated))

    # Update the transimission tree
    trans_tree = sim_data.transmission_tree()
    trans_tree_old = state.trans_tree
    if trans_tree_old is None:
        state.trans_tree = trans_tree
    else:
        base_t = state.t
        for eds in trans_tree.edges(data=True):
            tt = eds[2]["time"] + base_t
            trans_tree_old.add_edge(eds[0], eds[1], time=tt)
        state.trans_tree = trans_tree_old

    # Tick time

    return sim_data, state


def contact_tracing(G, state, p_s, p_t, p_r, sampling_node_attr=None):
    """
    Contact tracing

    Parameters
    ----------
    state : State
        The state object
    p_s : float
        Sampling probability for the infected individuals
    p_t : float
        Sampling probability for appending non-parent neighbors
    p_r : float
        Probability that the contact tracing is conducted
    sampling_node_attr: dict (optional)
        If given, we sample infected nodes with the attribute named the key and
        the value of the given dict

    Returns
    -------
    infected_sampled : list
        List of infected nodes to be sampled before tracing
    traced_contact : list
        List of tracted contacts
    """

    if sampling_node_attr is None:
        infected = state.infected
    else:
        k = list(sampling_node_attr.keys())[0]
        v = list(sampling_node_attr.values())[0]
        infected = [x for x in state.infected if G.nodes[x][k] == v]

    infected_sampled = list(
        np.random.choice(
            infected, np.random.binomial(len(infected), p_s), replace=False
        )
    )
    trans_tree = state.trans_tree
    traced_contact = Counter()
    for node in infected_sampled:
        if node in trans_tree and list(trans_tree.predecessors(node)):
            parent = list(trans_tree.predecessors(node))[0]
            temp = [
                x for x in G.neighbors(node) if x == parent or np.random.random() < p_t
            ]
        else:
            temp = [x for x in G.neighbors(node) if np.random.random() < p_t]
        for i in temp:
            if np.random.random() < p_r:
                traced_contact[i] += 1
    return infected_sampled, traced_contact


def eval_interv(
    load_network,
    rho,
    sir_params,
    t_interv,
    dt,
    interv,
    num_sim,
    infectable_node_attr=None,
):
    """
    Evaluate the efficacy of the intervention in question using the SIR model.
    We assume the SIR model for exponentially distributed transmission time
    and recovery time.

    Parameters
    ----------
    load_network : function
        A function that returns a networkx.Graph object
    rho : float
        fraction of initial infected individuals
    sir_params : dict
        parameters for the SIR simulation
    t_interv : float
        Time at which intervene the SIR process
    dt : float
        Time lag after at which we evaluate the efficacy of the intervention
    interv : function
        A function that returns nodes to be isolated.
        The function that takes the base network, G, and state variable as input.
    num_sim : dict
        Number of simulations
    infectable_node_attr: dict (optional)
        If given, we sample the initial infected nodes from those with the attribute
        with key and the value of the given dict (infectable nodes). Plus, we count
        the number of cases for only the infectable nodes.

    Returns
    -------
    num_infected: numpy.ndarray
        Number of infected nodes at time t_interv
    num_isolated: numpy.ndarray
        Number of isolated nodes at time t_interv
    num_new_cases: numpy.ndarray
        Number of new cases after time t_interv
    num_direct_prev_cases: numpy.ndarray
        Number of nodes that avoid infections due to the isolation by the intervention
    num_indirect_prev_cases: numpy.ndarray
        Number of nodes that avoid infections due to the isolation of other nodes by
        the intervention
    num_prev_cases: numpy.ndarray
        Number of nodes that avoid infections due to the intervention
    """

    def get_unique_offspr(node_set, trans_tree):
        offspr = [nx.descendants(trans_tree, x) for x in node_set if x in trans_tree]
        if len(offspr) == 0:
            return set([])
        else:
            return reduce(or_, offspr) - set(node_set)

    num_infected = np.zeros(num_sim)
    num_isolated = np.zeros(num_sim)
    num_new_cases = np.zeros(num_sim)
    num_prev_cases = np.zeros(num_sim)
    num_indirect_prev_cases = np.zeros(num_sim)
    num_direct_prev_cases = np.zeros(num_sim)
    num_nodes = np.zeros(num_sim)

    for exp_id in range(num_sim):
        # Load a network
        G = load_network()

        # Inialize the state
        state = State()
        state = state.init_state(G, rho, infectable_node_attr)

        # Run the SIR and generate the secondary cases
        sim_data, state = run_sir(G, t_interv, sir_params, state)

        state_bfr_int = copy.deepcopy(state)

        # Intervention
        isolated = interv(G, state)

        # Simulate SIR for dt
        sim_data, state = run_sir(G, dt, sir_params, state)

        # Find the cases due to infected nodes
        trans_tree = state.trans_tree
        offspr_from_infected = get_unique_offspr(state_bfr_int.infected, trans_tree)

        # Find the cases due to isolated nodes
        offspr_from_isolated = get_unique_offspr(isolated, trans_tree)

        # slice
        if infectable_node_attr is not None:
            k = list(infectable_node_attr.keys())[0]
            v = list(infectable_node_attr.values())[0]
            offspr_from_infected = set(
                [x for x in offspr_from_infected if G.nodes[x][k] == v]
            )
            offspr_from_isolated = set(
                [x for x in offspr_from_isolated if G.nodes[x][k] == v]
            )

        # Find the cases that can be prevented by intervations
        # Two cases:
        # (i) not infected due to being isolated before infection (direct prevention)
        # (ii) not infected due to the isolation for other nodes (indirected prevention)
        direct_prev_cases = offspr_from_infected & set(isolated)
        indirect_prev_cases = offspr_from_infected & offspr_from_isolated

        # Count the cases
        num_infected[exp_id] = len(
            state_bfr_int.infected
        )  # Infected nodes before intervention
        num_isolated[exp_id] = len(isolated)  # Isolated nodes
        num_new_cases[exp_id] = len(offspr_from_infected)  # cases from infected nodes
        num_direct_prev_cases[exp_id] = len(direct_prev_cases)
        num_indirect_prev_cases[exp_id] = len(indirect_prev_cases)
        num_prev_cases[exp_id] = len(
            direct_prev_cases | indirect_prev_cases
        )  # cases from isolated nodes
        num_nodes[exp_id] = G.number_of_nodes()

    return (
        num_infected,
        num_isolated,
        num_new_cases,
        num_direct_prev_cases,
        num_indirect_prev_cases,
        num_prev_cases,
        num_nodes,
    )
