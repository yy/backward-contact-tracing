import sys
from collections import Counter

import networkx as nx
import numpy as np

import graph_tool.all as gt
import pandas as pd
import seir
import utils_cont_ct as utils
from joblib import Parallel, delayed


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


def find_infected(sim_data, tree, tmin, tmax):

    retval = []
    for node in tree.nodes():
        node_history = sim_data.node_history(node)
        infected_time = np.inf
        recovery_time = np.inf
        for sid in range(len(node_history[1])):
            if node_history[1][sid] == "I":
                infected_time = node_history[0][sid]
            if node_history[1][sid] == "R":
                recovery_time = node_history[0][sid]
        if (infected_time <= tmin) and (tmax <= recovery_time):
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


def run_seir_simulations(G, num_samples, params, n_jobs=30):
    def _run_seir_simulations(G, params):
        sim_data = seir.run_SEIR(G, **params)
        tree = sim_data.transmission_tree()
        return {"tree": tree, "sim_data": sim_data}

    sim_data_list = Parallel(n_jobs=n_jobs)(
        delayed(_run_seir_simulations)(G, params) for i in range(num_samples)
    )
    return sim_data_list


def run_contact_tracing(G, sim_data, tree, p_s, p_t, n, interv_t, case_isolatable):

    infected_nodes = find_infected(sim_data, tree, interv_t, interv_t)
    infected_nodes = infected_nodes.intersection(case_isolatable)

    isolated_nodes = [x for x in infected_nodes if (np.random.rand() < p_s)]
    traced = trace_contacts(isolated_nodes, G, tree, p_t, n)

    parents = [utils.find_parent(x, tree) for x in isolated_nodes]
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
    gamma = float(sys.argv[2])
    frac_gather = float(sys.argv[3])
    E2I_rate = float(sys.argv[4])  # rate from e2i
    trans_rate = float(sys.argv[5])  # transmission rate
    recov_rate = float(sys.argv[6])  # recovery rate
    num_samples = int(sys.argv[7])  # number of samples
    p_s = float(sys.argv[8])  # detection probability
    p_t = float(sys.argv[9])  # trac time of intervention
    interv_t = float(sys.argv[10])  # transmission rate
    OUTPUT = sys.argv[11]

    min_deg_left = 2
    min_deg_right = 1
    ave_deg = 5

    params = {
        "E2I_rate": E2I_rate,
        "trans_rate": trans_rate,  # transmission rate (for edges)
        "recov_rate": recov_rate,  # recovery rate (for nodes)
        "init_seed_frac": 0.001,  # initial seed fraction
    }
    interv_params = {
        "p_s": p_s,
        "p_t": p_t,
        "interv_t": interv_t,
    }

    G = generate_power_law_bipartite_net(
        NET_SIZE,
        frac_gather,
        gamma,
        ave_deg,
        min_deg_left,
        min_deg_right,
        ["gathering", "people"],
    )
    case_isolatable = set([x for x in G.nodes if G.nodes[x]["class"] == "people"])
    params["initially_infected_nodes"] = [
        s for s in case_isolatable if np.random.rand() <= params["init_seed_frac"]
    ]

    sim_data_list = run_seir_simulations(G, num_samples, params)

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
                G, sim_data, tree, n=n, case_isolatable=case_isolatable, **interv_params
            )
            isolated_list += [isolated]
            parent_list += [parents]
            traced_list += [traced]

    isolated = pd.concat(isolated_list, ignore_index=True)
    traced = pd.concat(traced_list, ignore_index=True)
    parents = pd.concat(parent_list, ignore_index=True)

    deg = np.array(list(dict(G.degree()).values()))
    deg_people = np.array(
        [
            G.degree(x)
            for i, x in enumerate(G.nodes())
            if G.nodes[x]["class"] == "people" and deg[i] != 0
        ]
    )
    deg_gathering = np.array(
        [
            G.degree(x)
            for i, x in enumerate(G.nodes())
            if G.nodes[x]["class"] == "gathering" and deg[i] != 0
        ]
    )

    def get_Pk(a):
        u, freq = np.unique(a, return_counts=True)
        print(np.sum(freq))
        freq = freq / np.sum(freq)
        return dict(zip(u, freq))

    def get_Pk2(a):
        Nk = Counter(dict(G.degree()).values())
        Pk = {x: Nk[x] / float(G.order()) for x in Nk.keys()}
        return Pk

    def normalize_prob(ak):
        denom = sum([p for k, p in ak.items()])
        for k, p in ak.items():
            ak[k] /= denom
        return ak

    g0 = get_Pk(deg_people)
    f0 = get_Pk(deg_gathering)
    g1 = normalize_prob(dict((k - 1, k * p) for k, p in g0.items()))
    f1 = normalize_prob(dict((k - 1, k * p) for k, p in f0.items()))
    f2 = normalize_prob(dict((k - 2, k * (k - 1) * p) for k, p in f0.items()))

    pk_ccdf = ccdf_from_pk(g0)  # distribution for the people
    qk_ccdf = ccdf_from_pk(g1)  # distribution for the infected nodes
    rk_ccdf = ccdf_from_pk(f2)  # distribution for the back traced nodes

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
