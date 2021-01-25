"""Useful functions."""
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

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


def lineplot(
    df_list,
    data_params,
    plot_kwargs,
    xlabel,
    ylabel,
    ax,
    xticks=None,
    yticks=None,
    inset_bbox=None,
    inset_xticks=None,
    inset_yticks=None,
    inset_zoom_params={},
):
    """
    line plot with inset
    """
    for i in range(len(data_params)):
        sns.lineplot(data=df_list[i], **data_params[i], **plot_kwargs, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=["$%g$" % x for x in xticks])
        ax.set_xlim(left=np.min(xticks), right=np.max(xticks))

    if yticks is not None:
        ax.set(yticks=yticks, yticklabels=["$%g$" % y for y in yticks])
        ax.set_ylim(bottom=np.min(yticks), top=np.max(yticks))

    axins = None
    if inset_bbox is not None:
        # inset axes....
        axins = ax.inset_axes(inset_bbox)
        for i in range(len(data_params)):
            sns.lineplot(data=df_list[i], **data_params[i], **plot_kwargs, ax=axins)
        if inset_xticks is not None:
            axins.set(
                xticks=inset_xticks, xticklabels=["$%g$" % x for x in inset_xticks]
            )
        if inset_yticks is not None:
            axins.set(
                yticks=inset_yticks, yticklabels=["$%g$" % x for x in inset_yticks]
            )
        axins.set_xlim(left=np.min(inset_xticks), right=np.max(inset_xticks))
        axins.set_ylim(bottom=np.min(inset_yticks), top=np.max(inset_yticks))

        axins.legend().remove()
        axins.set_ylabel("")
        axins.set_xlabel("")
        # ax.indicate_inset_zoom(axins, **inset_zoom_params)
    else:
        axins = None

    return ax, axins


def get_xydata_from_ax(axes):
    captions = "abcdefghijklmnopqrstuvwxyz"
    dflist = []
    for i, ax in enumerate(axes):
        key = captions[i]
        df = _get_xydata_from_ax(ax)
        df["plot"] = key
        dflist += [df]
    return pd.concat(dflist, ignore_index=True)


def _get_xydata_from_ax(ax):
    lines = [i for i in ax.lines if i.get_label() != "_nolegend_"]
    lclist = [
        i
        for i in ax.get_children()
        if isinstance(i, matplotlib.collections.LineCollection)
    ]
    dflist = []
    for i, lc in enumerate(lclist):
        line = lines[i]
        lc = lclist[i]
        yerr = np.vstack([yerr[:, 1] for yerr in lc.get_segments()])
        x = line.get_xdata()
        y = line.get_ydata()
        label = line.get_label()
        if yerr.shape[0] != x.size:
            yerr = np.vstack([y, y]).T
        df = pd.DataFrame(
            {"x": x, "y": y, "ylow": yerr[:, 0], "yhigh": yerr[:, 1], "label": label}
        )
        dflist += [df]
    df = pd.concat(dflist, ignore_index=True)
    return df
