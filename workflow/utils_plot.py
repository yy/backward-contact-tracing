# Matplotlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def generate_plot_data_getter(data_table, **preset):
    def match(k, v):
        if (k in _data_table.columns) is False:
            print(k, " is not found in data")
            return np.zeros(_data_table.shape[0]) == 1
        else:
            if _data_table.dtypes[k] == "O":
                return _data_table[k] == v
            else:
                return np.isclose(_data_table[k], v)

    _data_table = data_table.copy()
    sl = None
    for k, v in preset.items():
        if sl is None:
            sl = match(k, v)
        else:
            sl = sl & match(k, v)

    if sl is not None:
        _data_table = _data_table.loc[sl, :]

    def get_plot_data(**args):
        sl = None
        for k, v in args.items():
            if sl is None:
                sl = match(k, v)
            else:
                sl = sl & match(k, v)
        df = _data_table.loc[sl, :]
        if "time" in df.columns:
            resol = 1
            # resol = 12 * 24
            df.loc[:, "time"] = df["time"].values / resol - df["start"].values
        return df

    return get_plot_data


def generate_palette(intervention="contact tracing"):

    cmap = sns.color_palette().as_hex()
    if intervention == "intervention":
        cmap_div = ["#EBB799", "#DD8452", "#994B1E"]
    elif intervention == "none":
        cmap_div = ["#4C72B0", "#78899F", "#E1A16C", "#DD8452"]
    else:
        # cmap_div = ["#432371", "#9f6976", "#faae7b"]
        cmap_div = ["#7fa5c9", "#4c71b0", "#29487d"]

    def palette(x):
        if isinstance(x, str):
            if x == "no intervention":
                return "#6d6d6d"
            elif x == "case isolation":
                return cmap[0]
            elif x == "contact tracing":
                return cmap[1]
        else:
            return cmap_div[x]

    return palette


def getlineplotdata(data, x, y, label, **params):
    df = data[[x, y]].copy()
    df["label"] = label
    return [df]


def plot_time_vs_i(data, ax, plot_params, maxnode, p_s=0.05, **data_params):

    plot_params["x"] = "time"
    plot_params["y"] = "Infected nodes"
    palette = generate_palette()
    get_plot_data = generate_plot_data_getter(data, **data_params)
    sns.lineplot(
        data=get_plot_data(intervention="no intervention"),
        label="No intervention",
        **plot_params,
        color=palette("no intervention"),
        ax=ax,
    )

    get_plot_data = generate_plot_data_getter(
        data, intervention="intervention", p_s=p_s, **data_params
    )
    palette = generate_palette("intervention")
    for i, p_t in enumerate([0, 0.5, 1.0]):
        if p_t < 1e-10:
            sns.lineplot(
                data=get_plot_data(p_t=p_t),
                label="Case isolation, $p_t=0$",
                **plot_params,
                color=palette(i),
                ax=ax,
            )
        else:
            sns.lineplot(
                data=get_plot_data(maxnode=maxnode, p_t=p_t),
                label="$p_t=%.1f$" % p_t,
                **plot_params,
                color=palette(i),
                ax=ax,
            )
    return ax


def plot_time_vs_i_maxnode(
    data, ax, plot_params, maxnode_list, linestyles, p_s=0.05, p_t=0.5, **data_params
):
    """
    maxnode_list : list
        int array
    linestyles : dict
        key : max node
        value : line style
    """

    plot_params["x"] = "time"
    plot_params["y"] = "Infected nodes"
    intervention = "contact tracing"

    cmap = generate_palette(intervention)
    get_plot_data = generate_plot_data_getter(
        data, intervention="intervention", p_s=p_s, p_t=p_t, **data_params
    )
    # for i, ps in enumerate([0.1, 0.52, 1]):
    # linestyles = {3: ":", 6: "--", 9999: "-"}
    # for i, maxnode in enumerate([3, 6, 9999]):
    for _, maxnode in enumerate(maxnode_list):

        if maxnode not in linestyles:
            continue

        df = get_plot_data(maxnode=maxnode)
        if maxnode < maxnode_list[-1]:
            sns.lineplot(
                data=df,
                **plot_params,
                color=cmap("contact tracing"),
                label="$n = %g$" % maxnode,
                ax=ax,
            )
        else:
            sns.lineplot(
                data=df,
                **plot_params,
                color=cmap("contact tracing"),
                linestyle=linestyles[maxnode],
                label="$n=$ all",
                ax=ax,
            )
        ax.lines[-1].set_linestyle(linestyles[maxnode])

    return ax


def plot_time_vs_i_ps(data, ax, plot_params, p_t=0.5, **data_params):
    """
    maxnode_list : list
        int array
    linestyles : dict
        key : max node
        value : line style
    """

    plot_params["x"] = "time"
    plot_params["y"] = "Infected nodes"

    cmap = generate_palette()

    get_plot_data = generate_plot_data_getter(
        data, intervention="intervention", p_t=p_t, **data_params
    )

    ps_list = [0.05, 0.25, 0.5]
    for i, ps in enumerate(ps_list):

        df = get_plot_data(ps=ps)
        sns.lineplot(
            data=df,
            **plot_params,
            color=cmap(i),
            label="$p_s = %.2f$" % ps if ps > 0 else "$p_s=0$",
            ax=ax,
        )

    return ax


def plot_ps_vs(data, ax, plot_params, maxnode, ls="-", **data_params):
    plot_params["x"] = "pt"
    cmap = generate_palette()

    for i, p_s in enumerate([0.05, 0.25, 0.5]):

        get_plot_data = generate_plot_data_getter(
            data, intervention="contact tracing", p_s=p_s, **data_params
        )

        sns.lineplot(
            data=get_plot_data(maxnode=maxnode).rename(columns={"p_t": "pt"}),
            # marker="o",
            label=r"$p_s=%g$" % p_s,
            **plot_params,
            color=cmap(i),
            ax=ax,
        )
        ax.lines[-1].set_linestyle(ls)

    xticks = np.linspace(0, 1, 5)
    ax.set(xticks=xticks, xticklabels=["$%g$" % x for x in xticks])
    return ax


def plot_ps_vs_i(data, ax, plot_params, maxnode, ls="-", **data_params):
    plot_params["y"] = "Isolated cases"
    return plot_ps_vs(data, ax, plot_params, maxnode, ls=ls, **data_params)


def plot_ps_vs_p(data, ax, plot_params, maxnode, ls="-", **data_params):
    plot_params["y"] = "Preventable cases"
    return plot_ps_vs(data, ax, plot_params, maxnode, ls=ls, **data_params)


def plot_ps_vs_e(data, ax, plot_params, maxnode, ls="-", **data_params):
    plot_params["y"] = "efficiency"
    return plot_ps_vs(data, ax, plot_params, maxnode, ls=ls, **data_params)
