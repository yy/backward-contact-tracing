import json
import sys
from collections import Counter

import EoN
import matplotlib.pyplot as plt
import networkx as nx

import seaborn as sns
import utils

FIG_FILE = sys.argv.pop()
T_CONTACT_TRACING = float(sys.argv.pop())
BASE_NET = sys.argv.pop()
RESULT_FILES = sys.argv[1:]


def aggregate_results(G, result_files):
    ccdfs = {}

    infected = []
    infected_sampled = []
    parents = []
    to_be_traced_all = []
    to_be_traced_25 = []
    to_be_traced_10 = []
    for resfile in result_files:
        with open(resfile, "r") as f:
            result = json.load(f)
        infected += [G.degree(x) for x in result["infected"]]
        infected_sampled += [G.degree(x) for x in result["infected_sampled"]]
        parents += [G.degree(x) for x in result["parents"]]
        to_be_traced = Counter(result["to_be_traced"])
        to_be_traced_all += [G.degree(x) for x in to_be_traced.keys()]
        to_be_traced_25 += [G.degree(x[0]) for x in to_be_traced.most_common(25)]
        to_be_traced_10 += [G.degree(x[0]) for x in to_be_traced.most_common(10)]

    ccdfs["infected"] = utils.ccdf_from_data(infected)
    ccdfs["parents"] = utils.ccdf_from_data(parents)
    ccdfs["traced"] = utils.ccdf_from_data(to_be_traced_all)
    ccdfs["traced25"] = utils.ccdf_from_data(to_be_traced_25)
    ccdfs["traced10"] = utils.ccdf_from_data(to_be_traced_10)
    return ccdfs


def calculate_ccdfs_from_network(G):
    avg_deg = EoN.get_PGFPrime(EoN.get_Pk(G))(1)
    rk_norm = EoN.get_PGFDPrime(EoN.get_Pk(G))(1)

    pk = dict(sorted(EoN.get_Pk(G).items()))
    qk = dict((k - 1, k * p / avg_deg) for k, p in pk.items())
    rk = dict((k - 2, k * (k - 1) * p / rk_norm) for k, p in pk.items())
    return dict(
        (name, ccdf)
        for name, ccdf in zip(
            ["pk", "qk", "rk"], list(map(utils.ccdf_from_pk, [pk, qk, rk]))
        )
    )


if __name__ == "__main__":
    G = nx.read_edgelist(BASE_NET)
    ccdfs = calculate_ccdfs_from_network(G)
    ccdfs.update(aggregate_results(G, RESULT_FILES))

    # Plot styles and figure objects
    sns.set_style("white")
    sns.set(font_scale=1)
    sns.set_style("ticks")

    cmap = sns.color_palette().as_hex()

    colors = {"pk": (0, 0, 0), "qk": (0.5, 0.5, 0.5), "rk": (0.7, 0.7, 0.7)}
    base_linewidth = 0.7
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 7), sharex=True)

    # Panel 1
    utils.loglog_plot(
        [ccdfs[x] for x in ["pk", "qk", "rk", "infected", "parents"]],
        kwargs_list=[
            {
                "label": r"$G_0(x)~ (\sim p_k$)",
                "color": colors["pk"],
                "linewidth": base_linewidth,
            },
            {
                "label": r"$G_1(x)~ (\sim k p_k$)",
                "color": colors["qk"],
                "linewidth": base_linewidth,
            },
            {
                "label": r"$G_2(x)~ (\sim k^2 p_k)$",
                "color": colors["rk"],
                "linewidth": base_linewidth,
            },
            {
                "label": r"Infected ($t={:.1f}$)".format(T_CONTACT_TRACING),
                "color": cmap[1],
            },
            {
                "label": r"Sampled parents ($t={:.1f}$)".format(T_CONTACT_TRACING),
                "color": cmap[0],
            },
        ],
        xlabel=None,
        ylabel=None,
        ax=ax1,
    )

    # Panel 2
    utils.loglog_plot(
        [
            ccdfs[x]
            for x in ["pk", "qk", "rk", "infected", "traced", "traced25", "traced10"]
        ],
        kwargs_list=[
            {"color": colors["pk"], "linewidth": base_linewidth},
            {"color": colors["qk"], "linewidth": base_linewidth},
            {"color": colors["rk"], "linewidth": base_linewidth},
            {"label": r"Infected".format(T_CONTACT_TRACING), "color": cmap[1]},
            {
                "label": r"Traced, all",
                "color": "{col}77".format(col=cmap[0]),
                "ls": ":",
            },
            {
                "label": r"Traced, top 25",
                "color": "{col}aa".format(col=cmap[0]),
                "ls": "--",
            },
            {
                "label": r"Traced, top 10",
                "color": "{col}ff".format(col=cmap[0]),
                "ls": "-",
            },
        ],
        xlabel=r"$k$",
        ylabel=None,
        ax=ax2,
    )
    ax2.set_xlabel(r"$k$", fontsize=20)

    # Labels
    fig.text(-0.05, 0.5, "CCDF", rotation=90, fontsize=16)

    # Labels
    ax1.annotate(
        "a",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=25,
    )
    ax2.annotate(
        "b",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=25,
    )

    plt.tight_layout()
    plt.savefig(FIG_FILE, dpi=300, bbox_inches="tight")
