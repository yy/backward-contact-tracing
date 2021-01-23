"""
Utility functions for simulation SIR model with intervention by contact tracing. 

Notes
-----
- We assume the SIR model for exponentially distributed infection and recovery times. 
- The utility functions process a `state` variable, which is represented by a python dict object (see `init_state` function)
- The `state` variable contains all the information needed to run the SIR simulations, contact tracing and isolations by the contact tracing. 
"""

from collections import Counter
import numpy as np
import networkx as nx
import copy
import EoN
from operator import or_
from functools import reduce


class State:
    def __init__(self):
        self.infected = []  # list of infected nodes
        self.recovered = []  # list of recovered nodes
        self.isolated = []  # list of isolated nodes by intervention
        self.trans_tree = None  # transmission tree
        self.t = 0  # time counter

    def init_state(self, G, rho):
        """
        Initialize the state object
        
        Parameters
        ----------
        G : networkx.Graph
            Base graph
        rho : float
            Fraction of initial infected individuals
            
        Returns
        -------
        state : State 
        """
        nodes = list(G.nodes())
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


def contact_tracing(G, state, p_s, p_t, p_r):
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
        
    Returns
    -------
    infected_sampled : list 
        List of infected nodes to be sampled before tracing 
    traced_contact : list 
        List of tracted contacts
    """
    infected = state.infected
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


def eval_interv(G, rho, sir_params, t_interv, dt, interv, num_sim):
    """
    Evaluate the efficacy of the intervention in question using the SIR model.
    We assume the SIR model for exponentially distributed transmission time and recovery time.  
    
    Parameters
    ----------
    G : networkx.Graph
        Base network
    rho : float
        fraction of initial infected individuals
    sir_params : dict
        parameters for the SIR simulation
    interv : function
        A function that returns nodes to be isolated. 
        The function that takes the base network, G, and state variable as input.
    t_interv : float 
        Time at which intervene the SIR process 
    dt : float
        Time lag after at which we evaluate the efficacy of the intervention
    num_sim : dict
        Number of simulations 
    
    Returns
    -------
    num_isolated : np.ndarray
        Number of isolated nodes
    frac_preventable_cases : nd.ndarray
        Fraction of preventable cases
    """
    num_infected = np.zeros(num_sim)
    num_isolated = np.zeros(num_sim)
    num_new_cases = np.zeros(num_sim)
    num_prev_cases = np.zeros(num_sim)
    num_indirect_prev_cases = np.zeros(num_sim)
    num_direct_prev_cases = np.zeros(num_sim)

    for exp_id in range(num_sim):

        # Inialize the state
        state = State()
        state = state.init_state(G, rho)

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

    return (
        num_infected,
        num_isolated,
        num_new_cases,
        num_direct_prev_cases,
        num_indirect_prev_cases,
        num_prev_cases,
    )


def get_unique_offspr(node_set, trans_tree):
    offspr = [nx.descendants(trans_tree, x) for x in node_set if x in trans_tree]
    if len(offspr) == 0:
        return set([])
    else:
        return reduce(or_, offspr) - set(node_set)
