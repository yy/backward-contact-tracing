import sys

import EoN
import networkx as nx
import numpy as np

import pandas as pd


def construct_transmission_tree(sim_logs):
    tree_list = []
    for sim_id, sim_log in sim_logs.groupby("id"):
        se_event = sim_log[sim_log.type == "i"].dropna()
        se_event["source"] = se_event["source"].astype(int)
        se_event["node"] = se_event["node"].astype(int)
        tree = nx.from_pandas_edgelist(
            df=se_event,
            source="source",
            target="node",
            edge_attr="elapsed",
            create_using=nx.DiGraph,
        )

        # Find the root of the tree
        first_cases = [n for n, d in tree.in_degree() if d == 0]
        infected_time = dict(zip(first_cases, np.zeros(len(first_cases))))
        nx.set_node_attributes(tree, infected_time, "infected_time")

        # Set the time of infection as an attribute of each node
        infected_time = sim_log[sim_log.type == "i"][["node", "elapsed"]]
        infected_time = dict(zip(infected_time.node, infected_time.elapsed))
        nx.set_node_attributes(tree, infected_time, "infected_time")

        # Set the time of recovery as an attribute of each node
        recovery_time = sim_log[sim_log.type == "r"][["node", "elapsed"]]
        recovery_time = dict(zip(recovery_time.node, recovery_time.elapsed))
        nx.set_node_attributes(tree, recovery_time, "recovery_time")

        # Add three to the list
        tree_list += [tree]
    return tree_list


def find_parent(node, trans_tree):
    if node in trans_tree and list(trans_tree.predecessors(node)):
        parent = list(trans_tree.predecessors(node))[0]
        return parent
    else:
        return None


def trace_contacts(
    node_list, G, trans_tree, p_t, n, already_isolated=set([]),
):
    contact_list = []
    for node in node_list:
        contacts = []
        for neighbor in G[node]:
            if neighbor in already_isolated:
                continue
            if np.random.random_sample() < p_t:
                contacts += [int(neighbor)]
        contact_list += contacts

    if len(contact_list) > 0:
        ctrace_isolated, freq = np.unique(contact_list, return_counts=True)
        ids = np.argsort(freq)[::-1][: np.minimum(n, len(ctrace_isolated))]
        return set(ctrace_isolated[ids])
    else:
        return set([])


def find_cases(tree, t_min):
    return [x[0] for x in tree.nodes(data=True) if t_min < x[1]["infected_time"]]


def find_infected(sim_data, tree, tmin, tmax):

    retval = []
    for node in tree.nodes():
        node_history = sim_data.node_history(node)
        infected_time = np.inf
        recovery_time = np.inf
        for sid in range(len(node_history)):
            if node_history[1][sid] == "I":
                infected_time = node_history[0][sid]
            if node_history[1][sid] == "R":
                recovery_time = node_history[0][sid]
        if (infected_time <= tmin) and (tmax <= recovery_time):
            # if (infected_time<=tmin) and (tmax<=recovery_time):
            retval += [node]
    return set(retval)


def ccdf_from_pk(pk):
    """Calculate CCDF from pk."""
    Y = np.flip(np.cumsum(np.flip(np.array(list(pk.values())))))
    X = list(pk.keys())
    return pd.DataFrame({"prob": Y, "deg": X})


def ccdf_from_data(data):
    """Calculate CCDF from 1-d data."""
    if len(data) == 0:
        return ([], [])
    data, freq = np.unique(data, return_counts=True)
    return ccdf_from_pk(dict(zip(data, freq / np.sum(freq))))


def run_sir_simulations(G, num_samples, params):
    sim_data_list = []
    for i in range(num_samples):
        sim_data = EoN.fast_SIR(G, return_full_data=True, **params)
        tree = sim_data.transmission_tree()
        sim_data_list += [{"tree": tree, "sim_data": sim_data}]
    return sim_data_list


def run_contact_tracing(G, sim_data, tree, p_s, p_t, n, interv_t):

    infected_nodes = find_infected(sim_data, tree, interv_t, interv_t)

    isolated_nodes = [x for x in infected_nodes if (np.random.rand() < p_s)]

    traced = trace_contacts(isolated_nodes, G, tree, p_t, n)

    parents = [find_parent(x, tree) for x in isolated_nodes]
    parents = [x for x in parents if x is not None]

    deg = G.degree()
    isolated = pd.DataFrame(
        [{"node": x, "deg": deg[x], "n": n} for x in isolated_nodes]
    )
    traced = pd.DataFrame([{"node": x, "deg": deg[x], "n": n} for x in traced])
    parents = pd.DataFrame([{"node": x, "deg": deg[x], "n": n} for x in parents])

    return isolated, parents, traced


if __name__ == "__main__":

    NET_SIZE = int(sys.argv[1])  # network size
    M = int(sys.argv[2])  # min degree
    trans_rate = float(sys.argv[3])  # transmission rate
    recov_rate = float(sys.argv[4])  # recovery rate
    num_samples = int(sys.argv[5])  # number of samples
    p_s = float(sys.argv[6])  # detection probability
    p_t = float(sys.argv[7])  # trac time of intervention
    interv_t = float(sys.argv[8])  # transmission rate
    OUTPUT = sys.argv[9]

    params = {
        "tmax": 100,
        "tau": trans_rate,  # transmission rate (for edges)
        "gamma": recov_rate,  # recovery rate (for nodes)
        "rho": 0.001,  # initial seed fraction
    }
    interv_params = {
        "p_s": p_s,
        "p_t": p_t,
        "interv_t": interv_t,
    }

    G = nx.barabasi_albert_graph(NET_SIZE, M)
    sim_data_list = run_sir_simulations(G, num_samples, params)

    # Construct the transmission tree
    nlist = [1, 10, 50, 99999999999]

    isolated_list = []
    traced_list = []
    parent_list = []
    infected_list = []
    for n in nlist:

        for gid in range(len(sim_data_list)):

            sim_data = sim_data_list[gid]["sim_data"]
            tree = sim_data_list[gid]["tree"]
            isolated, parents, traced = run_contact_tracing(
                G, sim_data, tree, n=n, **interv_params
            )
            isolated_list += [isolated]
            parent_list += [parents]
            traced_list += [traced]

    isolated = pd.concat(isolated_list, ignore_index=True)
    traced = pd.concat(traced_list, ignore_index=True)
    parents = pd.concat(parent_list, ignore_index=True)

    avg_deg = np.mean(list(G.degree), axis=0)[1]
    rk_norm = EoN.get_PGFDPrime(EoN.get_Pk(G))(1)

    pk = dict(sorted(EoN.get_Pk(G).items()))
    qk = dict((k - 1, k * p / avg_deg) for k, p in pk.items())
    rk = dict((k - 2, k * (k - 1) * p / rk_norm) for k, p in pk.items())

    pk_ccdf = ccdf_from_pk(pk)
    qk_ccdf = ccdf_from_pk(qk)
    rk_ccdf = ccdf_from_pk(rk)

    nmax = parents["n"].max()
    parents_ccdf = ccdf_from_data(parents[parents["n"] == nmax]["deg"].values)
    isolated_ccdf = ccdf_from_data(isolated[isolated["n"] == nmax]["deg"].values)
    traced_n_ccdf = {}
    for n in nlist:
        traced_n_ccdf[n] = ccdf_from_data(traced[traced["n"] == n]["deg"].values)

    pk_ccdf["type"] = "pk"
    qk_ccdf["type"] = "qk"
    rk_ccdf["type"] = "rk"

    parents_ccdf["type"] = "parents"
    isolated_ccdf["type"] = "isolated"

    tables = [pk_ccdf, qk_ccdf, rk_ccdf, parents_ccdf, isolated_ccdf]
    for n in nlist:
        traced_n_ccdf[n]["type"] = "traced_%d" % n
        tables += [traced_n_ccdf[n]]

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUTPUT, sep="\t", index=False)
