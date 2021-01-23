"""Useful functions."""
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["text.usetex"] = True


def ccdf_from_pk(pk):
    """Calculate CCDF from pk."""
    Y = np.flip(np.cumsum(np.flip(np.array(list(pk.values())))))
    X = list(pk.keys())
    return X, Y


def ccdf_from_data(data):
    """Calculate CCDF from 1-d data."""
    if not data:
        return ([], [])
    N = len(data)
    Y = np.linspace(1.0, 1 / N, num=N)
    X = sorted(data)
    return X, Y


def loglog_plot(
    xy_pair_list,
    kwargs_list=None,
    figsize=(3.5, 2.8),
    xlabel="k",
    ylabel="p(k)",
    ax=None,
):
    """Draw loglog plots for the plot data (xy_pair_list)."""
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if kwargs_list:
        for plot_data_pair, kwargs in zip(xy_pair_list, kwargs_list):
            ax.loglog(*plot_data_pair, **kwargs)
        ax.legend(frameon=False)
    else:
        for plot_data_pair in xy_pair_list:
            ax.loglog(*plot_data_pair)
    if not ax:
        return fig


def plot_pk(pk, xlabel="k", ylabel="p(k)"):
    """Plot the degree distribution."""
    return loglog_plot([zip(*sorted(pk.items()))], xlabel=xlabel, ylabel=ylabel)


def plot_ccdf_from_pk(pk, xlabel="k", ylabel="CCDF"):
    """Plot CCDF using pk."""
    return loglog_plot([ccdf_from_pk(pk)], xlabel=xlabel, ylabel=ylabel)


def nodes_with_given_state_at_time(t, s, sim_data, node_set):
    """Return a set of infected nodes, out of `nodes` provided."""
    return [
        node for node, state in sim_data.get_statuses(node_set, t).items() if state == s
    ]


def trace_contacts(t, sim_data, G, p_sample, p_neighbor):
    """Perform contact tracing.

    Sample p_sample fraction of infected nodes at time t, and then trace
    some of the neighbors of these sampled nodes.
    """
    infected_sample = sample_infected(t, p_sample, sim_data, G.nodes())

    cnt = Counter()
    for node in infected_sample:
        for neighbor in G[node]:
            if np.random.random_sample() < p_neighbor:
                cnt[neighbor] += 1
    return cnt


def sample_infected(t, p_sample, sim_data, node_set):
    """Sample p_sample fraction of nodes from infected people at time t."""
    infected = nodes_with_given_state_at_time(t, "I", sim_data, node_set)
    return np.random.choice(infected, int(len(infected) * p_sample))
