import logging
import os
import sys

import networkx as nx
import numpy as np

import pandas as pd
import utils_cont_ct as utils

logging.basicConfig(level=logging.DEBUG)


def make_get_contact_list_func(G, trans_tree):
    """
    Parameters
    ----------
    G : networkx.Graph
        Base net
    trans_tree : networkx.DiGraph
        Transmission tree
    Returns
    -------
    func : func
        func(i, t) returns the contact list at time t for node i.
    """

    def get_recent_close_contacts(node, t):
        contacts = []
        for neighbor in G[node]:
            contacts += [int(neighbor)]
        contacts = np.array(contacts)
        # contacts = contacts[np.random.rand(contacts.size) > p_drop_out]
        return contacts.tolist()

    return get_recent_close_contacts


if __name__ == "__main__":

    # Input
    edgelist = sys.argv[1]
    sim_log_data = sys.argv[2]
    p_s = float(sys.argv[3])  # detection probability
    max_traced_nodes = int(sys.argv[4])  # number of nodes to isolate
    start_t = float(sys.argv[5])  # the time from which the intervention starts
    cycle_dt = float(sys.argv[6])  # interval for the contact tracing
    memory_dt = float(sys.argv[7])  # interval for the contact tracing
    time_lag_for_isolation = float(sys.argv[8])
    trace_mode = sys.argv[9]
    isolatable_node_type = None

    # Output
    OUTPUT_EVENT = sys.argv.pop()
    OUTPUT = sys.argv.pop()

    if len(sys.argv) == 12:
        isolatable_node_type = sys.pop()

    # Load data
    logging.debug("Loading data")
    filename, file_extension = os.path.splitext(edgelist)
    print(filename, file_extension)
    if file_extension == ".gexf":  # when a node has attributes
        G = nx.read_gexf(edgelist, node_type=int)
    elif file_extension == ".edgelist":  # when a node does not have attrbutes
        G = nx.read_edgelist(edgelist, nodetype=int)
    else:
        raise ValueError("The input graph should be saved in .edgelist or .gexf format")

    logs = pd.read_csv(sim_log_data)

    #
    # Preprocess
    #
    logging.debug("Construct the transmission tree from the log")
    logs["id"] = "id"
    tree_list = utils.construct_transmission_tree(logs)

    logging.debug("Set onset time")
    for tid, tree in enumerate(tree_list):
        tree_list[tid] = utils.set_onset_time(tree, time_lag_for_isolation)

    logging.debug("Generate the contact list func")
    get_contact_list = make_get_contact_list_func(G, tree_list[0])

    # Find people node
    if isolatable_node_type is not None:
        case_isolatable = set(
            [x for x in G.nodes if G.nodes[x]["class"] == isolatable_node_type]
        )
    else:
        case_isolatable = None

    #
    # Simulation
    #
    result_table, event_table = utils.simulate(
        tree_list,
        start_t,
        cycle_dt,
        memory_dt,
        get_contact_list,
        p_s,
        max_traced_nodes,
        trace_mode,
        case_isolatable,
    )

    result_table.to_csv(OUTPUT, sep="\t", compression="gzip")
    event_table.to_csv(OUTPUT_EVENT, sep="\t", compression="gzip")
