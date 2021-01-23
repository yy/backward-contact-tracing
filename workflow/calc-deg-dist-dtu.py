import logging
import sys

import EoN
import numpy as np
from scipy import sparse

import pandas as pd
import utils_cont_ct as utils

logging.basicConfig(level=logging.DEBUG)


def make_get_contact_list_func(contact_history, time_window, close_contact_threshold):
    """
    Parameters
    ----------
    contact_history:
        History of contacts. See utils.make_contact_history(sim_logs, contact_logs)
    time_window: int
        We will make a list of contacts made between (t-time_window,t]
    close_contact_threshold: int
        If a node has a contact with another node more than
        ``close_contact_threshold'', we consider the contact a
        a close contact. Other contacts will be dropped from the
        return. Set = 0 to retrieve all recent contacts.

    Returns
    -------
    func : func
        func(i, t) returns the contact list at time t for node i.
    """

    def get_recent_close_contacts(node_id, t):
        contacts = contact_history.get(node_id, None)

        if contacts is None:
            return []

        recent_contacts = contacts[
            ((t - time_window) < contacts["timestamp"]) * (contacts["timestamp"] <= t)
        ]
        freq = recent_contacts["contact"].value_counts()
        return [k for k, v in freq[freq > close_contact_threshold].items()]

    return get_recent_close_contacts


def get_infected_dist(tree, ps, dt, top_list, get_contact_list, get_degree):
    case_isolated = {
        x: tree.nodes[x]["onset_time"] for x in tree.nodes() if np.random.rand() <= ps
    }
    nodes = np.array(list(case_isolated.keys()))
    parents = [list(tree.predecessors(x)) for x in nodes]
    parents = [x[0] for x in parents if len(x) > 0]

    event_time = np.array(list(case_isolated.values()))
    group_ids = np.ceil(event_time / dt)

    ct_isolated = {k: [] for k in top_list}
    for group_id in np.sort(np.unique(group_ids)):
        # Get the contact list and concatenate them
        starting_nodes = nodes[group_id == group_ids]
        traced = [get_contact_list(x, group_id * dt) for x in starting_nodes]
        traced = sum(traced, [])
        if len(traced) > 0:
            traced, freq = np.unique(traced, return_counts=True)
            node_order = np.argsort(freq)[::-1]
            for k in top_list:
                if k <= len(traced):
                    ct_isolated[k] = ct_isolated[k] + traced[node_order[:k]].tolist()
                else:
                    ct_isolated[k] = ct_isolated[k] + traced.tolist()

    deg_samp = [get_degree(x) for x in nodes]
    deg_parent = [get_degree(x) for x in parents]

    deg_traced = {}
    for k in top_list:
        deg_traced[k] = [get_degree(x) for x in ct_isolated[k]]
        # deg_traced[k] = [get_degree(x) for x in ct_isolated[k] if x in tree]
    return deg_samp, deg_parent, deg_traced


def pk_from_data(data):
    data, freq = np.unique(data, return_counts=True)
    return dict(zip(data.astype(int), freq / np.sum(freq)))


def ccdf_from_pk(pk):
    """Calculate CCDF from pk."""
    Y = np.flip(np.cumsum(np.flip(np.array(list(pk.values())))))
    X = list(pk.keys())
    return pd.DataFrame({"prob": Y, "deg": X})


def ccdf_from_data(data):
    """Calculate CCDF from 1-d data."""
    if len(data) == 0:
        return ([], [])
    pk = pk_from_data(data)
    return ccdf_from_pk(pk)


def make_degree_distribution(contact_logs, dt_slice=12):
    # contact_logs["tid"] = np.floor(contact_logs["timestamp"] / dt_slice)
    deg_dist = {}
    deg_list = []
    N = int(np.max(contact_logs[["n1", "n2"]].values)) + 1
    A = sparse.csr_matrix(
        (np.ones(contact_logs.shape[0]), (contact_logs["n1"], contact_logs["n2"])),
        shape=(N, N),
    )
    A.data[A.data < dt_slice] = 0

    A = A + A.T
    A.data = np.ones_like(A.data)

    deg = np.array(A.sum(axis=0)).reshape(-1)
    node_ids = np.where(deg)[0]
    for node_id in node_ids:
        deg_dist[node_id] = deg[node_id]
        deg_list += [deg[node_id]]

    def get_degree(node):
        return deg_dist.get(node, 0)

    return get_degree, deg_list


if __name__ == "__main__":
    contact_data = sys.argv[1]
    sim_log_data = sys.argv[2]
    sim_meta_data = sys.argv[3]
    trace_time_window_day = float(sys.argv[4])  # time window for tracing contact
    close_contact_hour_per_day = float(sys.argv[5])
    ps = float(sys.argv[6])  # detection probability
    cycle_dt = float(sys.argv[7])  # above which we carry out the contact tracing
    time_lag_for_isolation = float(sys.argv[8])
    resol = float(sys.argv[9])
    OUTPUT = sys.argv[10]

    top_list = [3, 10, 9999]

    #
    # Recale the time scale
    #
    trace_time_window = trace_time_window_day * 24 * 12
    close_contact_threshold = trace_time_window_day * close_contact_hour_per_day * 12
    time_lag_for_isolation = time_lag_for_isolation * 12 * 24
    cycle_dt = cycle_dt * 12 * 24

    #
    # Load data
    #
    sim_logs = pd.read_csv(sim_log_data, sep=",")
    contact_logs = pd.read_csv(contact_data, sep=",")
    sim_meta_data = pd.read_csv(sim_meta_data, sep=",")

    #
    # Preprocess
    #
    logging.debug("Extract off set time")
    t_offset = dict(zip(sim_meta_data["id"].values, sim_meta_data["t"].values))
    get_degree, deg_list = make_degree_distribution(contact_logs, dt_slice=resol)

    #    tree_list = utils.construct_transmission_tree(sim_logs)
    #    contact_history = utils.make_contact_history(sim_logs, contact_logs)

    #   get_contact_list = make_get_contact_list_func(
    #       contact_history, trace_time_window, close_contact_threshold
    #   )

    # Sample degree
    deg_samp_list = []
    deg_parent_list = []
    deg_traced_list = {k: [] for k in top_list}
    for sim_id, log_file in sim_logs.groupby("id"):
        """
        Note:
        sim_log timestamp is an elapse time, time since the first infection.
        On the other hand, the timestamp in the contact data is the actual time.
        Thus, the alignment is necessary. Here, I convert the timestamp to the
        elapse time by offsetting.
        """
        t_0 = t_offset.get(sim_id)

        logging.debug("Make contact list")
        contact_history = utils.make_contact_history(
            log_file, contact_logs, t_offset=t_0
        )

        logging.debug("Make transmission tree")
        tree = utils.construct_transmission_tree(log_file)[0]

        logging.debug("Generate the contact list func")
        get_contact_list = make_get_contact_list_func(
            contact_history, trace_time_window, close_contact_threshold
        )

        ds, dp, dt = get_infected_dist(
            tree, ps, cycle_dt, top_list, get_contact_list, get_degree
        )
        deg_samp_list += ds
        deg_parent_list += dp
        for l in top_list:
            deg_traced_list[l] += dt[l]
    deg_samp = np.array(deg_samp_list)
    deg_parent = np.array(deg_parent_list)
    deg_traced = {}
    for k in top_list:
        deg_traced[k] = np.array(deg_traced_list[k])

    # To ccdf
    ccdf_samp = ccdf_from_data(deg_samp)
    ccdf_parent = ccdf_from_data(deg_parent)
    ccdf_traced = {}
    for k in top_list:
        ccdf_traced[k] = ccdf_from_data(deg_traced[k])

    ccdf_samp["type"] = "infected"
    ccdf_parent["type"] = "parent"
    tables = [ccdf_samp, ccdf_parent]
    for k, tb in ccdf_traced.items():
        tb["type"] = "traced_%d" % k
        tables += [ccdf_traced[k]]

    pk = pk_from_data(deg_list)

    avg_deg = np.mean(deg_list)
    rk_norm = EoN.get_PGFDPrime(pk)(1)

    qk = dict((k - 1, k * p / avg_deg) for k, p in pk.items())
    rk = dict((k - 2, k * (k - 1) * p / rk_norm) for k, p in pk.items())

    pk_ccdf = ccdf_from_pk(pk)
    qk_ccdf = ccdf_from_pk(qk)
    rk_ccdf = ccdf_from_pk(rk)
    pk_ccdf["type"] = "pk"
    qk_ccdf["type"] = "qk"
    rk_ccdf["type"] = "rk"

    tables += [pk_ccdf, qk_ccdf, rk_ccdf]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUTPUT, sep="\t", index=False)
