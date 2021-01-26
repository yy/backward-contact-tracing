from collections import defaultdict

import EoN
import networkx as nx
import numpy as np

import pandas as pd


def run_SEIR(
    G, E2I_rate, trans_rate, recov_rate, init_seed_frac, initially_infected_nodes=None
):

    N = G.number_of_nodes()

    H = nx.DiGraph()
    H.add_node("S")
    H.add_edge("E", "I", rate=E2I_rate)
    H.add_edge("I", "R", rate=recov_rate)

    J = nx.DiGraph()
    J.add_edge(("I", "S"), ("I", "E"), rate=trans_rate)
    IC = defaultdict(lambda: "S")

    if initially_infected_nodes is None:
        initially_infected_nodes = np.random.choice(
            N, np.round(N * init_seed_frac).astype(int), replace=False
        )
    for node in initially_infected_nodes:
        IC[node] = "E"

    return_statuses = ("S", "E", "I", "R")

    sim_data = EoN.Gillespie_simple_contagion(
        G,
        H,
        J,
        IC,
        return_full_data=True,
        return_statuses=return_statuses,
        tmax=float("Inf"),
    )
    return sim_data


def to_log(G, sim_data, filename=None):
    df = pd.DataFrame(sim_data.transmissions(), columns=["elapsed", "source", "node"])
    df["type"] = "e"

    N = G.number_of_nodes()
    df_sup = []
    for node in range(N):
        t, states = sim_data.node_history(node)
        dg = pd.DataFrame({"elapsed": t, "type": states, "node": node, "source": ""})
        df_sup += [dg]

    df_sup = pd.concat(df_sup, ignore_index=True)
    df_sup["type"] = df_sup["type"].str.lower()
    df_sup = df_sup[np.isin(df_sup["type"], ["i", "r"])]
    df = pd.concat([df, df_sup], ignore_index=True)

    if filename is not None:
        df.to_csv(filename, sep=",", index=False)

    return df
