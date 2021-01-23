import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import pandas as pd
import seaborn as sns
import utils

rc("text", usetex=True)

INT_PLOT_DATA = sys.argv[1]
INTB_PLOT_DATA = sys.argv[2]
FIG_FILE = sys.argv[3]

if __name__ == "__main__":

    def split_with_without_contact_tracing(df, p_r):
        return df[df.p_r == p_r], df[df.p_r == 0]

    p_r = 0.5  # prob. for conducting contact tracing
    frac_gathr_list = [0.25, 0.1]  # fraction of gathering nodes

    df_int = pd.read_csv(INT_PLOT_DATA, sep="\t")
    df_intb = pd.read_csv(INTB_PLOT_DATA, sep="\t")

    # Split the results into those for case isolation and contact tracing
    data4plot_int = split_with_without_contact_tracing(df_int, p_r)
    data4plot_intb = []
    for frac_gathr in frac_gathr_list:
        df = df_intb[df_intb.frac_gathr == frac_gathr]
        data4plot_intb += [split_with_without_contact_tracing(df, p_r)]

    #
    # Set parameters for plot
    #
    sns.set(font_scale=1)
    sns.set_style("white")
    sns.set_style("ticks")

    cmap = sns.color_palette()
    plot_kwargs = {"ci": 95, "err_style": "band"}

    # Panels a and d
    data_isolated_cases_int = {
        "data_params": [
            {
                "x": "p_s",
                "y": "f_num_isolated",
                "color": cmap[0],
                "label": "Contact tracing",
                "marker": "s",
                "markersize": 4,
            },
            {
                "x": r"p_s",
                "y": "f_num_isolated",
                "color": cmap[1],
                "label": "Case isolation",
                "marker": "o",
                "markersize": 4,
            },
        ],
        "xlabel": r"$p_s$",
        "ylabel": "Fraction of isolated cases",
    }
    ticks_params_isolated_cases_int = [
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 0.2, 6),
            "inset_bbox": None,
            # "inset_bbox": [0.12, 0.47, 0.32, 0.32],
            "inset_xticks": np.linspace(0, 0.1, 3),
            "inset_yticks": np.linspace(0, 0.01, 2),
        },
    ]
    # Panels b and e
    data_phi_int = {
        "data_params": [
            {
                "x": r"p_s",
                "y": r"f_num_prev_cases",
                "color": cmap[0],
                "marker": "s",
                "markersize": 4,
            },
            {
                "x": r"p_s",
                "y": r"f_num_prev_cases",
                "color": cmap[1],
                "marker": "o",
                "markersize": 4,
            },
        ],
        "xlabel": r"$p_s$",
        "ylabel": r"$\phi(t, t+\Delta)$",
    }
    ticks_params_phi_int = [
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 1.0, 6).tolist(),
            "inset_bbox": [0.5, 0.2, 0.43, 0.43],
            "inset_xticks": np.linspace(0, 0.1, 3),
            "inset_yticks": np.linspace(0, 1, 3),
        },
    ]
    # Panels c and f
    data_eff_int = {
        "data_params": [
            {
                "x": r"p_s",
                "y": r"protected_per_isolated",
                "color": cmap[0],
                "marker": "s",
                "markersize": 4,
            },
            {
                "x": r"p_s",
                "y": r"protected_per_isolated",
                "color": cmap[1],
                "marker": "o",
                "markersize": 4,
            },
        ],
        "xlabel": r"$p_s$",
        "ylabel": "Number of prevented cases \n per isolated case",
    }
    ticks_params_eff_int = [
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 15, 6),
            "inset_bbox": [0.45, 0.42, 0.5, 0.5],
            "inset_xticks": np.array([0.01, 0.05, 0.1]),
            "inset_yticks": np.linspace(0, 15, 4),
        },
    ]

    # Panels a and d
    data_isolated_cases_intb = {
        "data_params": [
            {
                "x": "p_s",
                "y": "f_num_isolated",
                "color": cmap[0],
                "label": "Contact tracing",
                "marker": "s",
                "markersize": 4,
            },
            {
                "x": r"p_s",
                "y": "f_num_isolated",
                "color": cmap[1],
                "label": "Case isolation",
                "marker": "o",
                "markersize": 4,
            },
        ],
        "xlabel": r"$p_s$",
        "ylabel": "Fraction of isolated cases",
    }
    ticks_params_isolated_cases_intb = [
        {
            "xticks": np.linspace(0, 1, 6),
            # "yticks": np.linspace(0, 0.1, 6),
            # "yticks": np.linspace(0, 0.15, 6),
            "yticks": np.linspace(0, 0.2, 6),
            "inset_bbox": None,
            # "inset_bbox": [0.16, 0.55, 0.28, 0.28],
            "inset_xticks": np.linspace(0, 0.1, 3),
            "inset_yticks": np.linspace(0, 0.006, 2),
        },
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 0.1, 6),
            "inset_bbox": None,
            # "inset_bbox": [0.23, 0.65, 0.28, 0.28],
            "inset_xticks": np.linspace(0, 0.1, 3),
            "inset_yticks": np.linspace(0, 0.01, 2),
        },
    ]
    # Panels b and e
    data_phi_intb = {
        "data_params": [
            {
                "x": r"p_s",
                "y": r"f_num_prev_cases",
                "color": cmap[0],
                "marker": "s",
                "markersize": 4,
            },
            {
                "x": r"p_s",
                "y": r"f_num_prev_cases",
                "color": cmap[1],
                "marker": "o",
                "markersize": 4,
            },
        ],
        "xlabel": r"$p_s$",
        "ylabel": r"$\phi(t, t+\Delta)$",
    }
    ticks_params_phi_intb = [
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 1.0, 6),
            "inset_bbox": None,
            # "inset_bbox": [0.68, 0.15, 0.28, 0.30],
            "inset_xticks": np.linspace(0, 0.1, 2),
            "inset_yticks": np.linspace(0, 0.5, 2),
        },
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 1.0, 6),
            "inset_bbox": None,
            # "inset_bbox": [0.68, 0.15, 0.28, 0.30],
            "inset_xticks": np.linspace(0, 0.1, 2),
            "inset_yticks": np.linspace(0, 0.5, 2),
        },
    ]
    # Panels c and f
    data_eff_intb = {
        "data_params": [
            {
                "x": r"p_s",
                "y": r"protected_per_isolated",
                "color": cmap[0],
                "marker": "s",
                "markersize": 4,
            },
            {
                "x": r"p_s",
                "y": r"protected_per_isolated",
                "color": cmap[1],
                "marker": "o",
                "markersize": 4,
            },
        ],
        "xlabel": r"$p_s$",
        "ylabel": "Number of prevented cases \n per isolated case",
    }
    ticks_params_eff_intb = [
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 15, 6),
            "inset_bbox": [0.35, 0.42, 0.5, 0.5],
            "inset_xticks": np.array([0.01, 0.05, 0.1]),
            "inset_yticks": np.linspace(0, 15, 4),
        },
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 15, 6),
            "inset_bbox": [0.35, 0.42, 0.5, 0.5],
            "inset_xticks": np.array([0.01, 0.05, 0.1]),
            "inset_yticks": np.linspace(0, 15, 4),
        },
    ]

    #
    # Plot the figure
    #
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10.5, 6.5), sharex=True)

    # Panels a and d
    utils.lineplot(
        data4plot_int,
        **data_isolated_cases_int,
        **ticks_params_isolated_cases_int[0],
        ax=axes[0, 0],
        plot_kwargs=plot_kwargs
    )

    # Panels b and e
    utils.lineplot(
        data4plot_int,
        **data_phi_int,
        **ticks_params_phi_int[0],
        ax=axes[1, 0],
        plot_kwargs=plot_kwargs
    )

    # Panels c and f
    utils.lineplot(
        data4plot_int,
        **data_eff_int,
        **ticks_params_eff_int[0],
        ax=axes[2, 0],
        plot_kwargs=plot_kwargs
    )

    for i, data in enumerate(data4plot_intb):
        # Panels a and d
        utils.lineplot(
            data,
            **data_isolated_cases_intb,
            **ticks_params_isolated_cases_intb[i],
            ax=axes[0, i + 1],
            plot_kwargs=plot_kwargs
        )

        # Panels b and e
        utils.lineplot(
            data,
            **data_phi_intb,
            **ticks_params_phi_intb[i],
            ax=axes[1, i + 1],
            plot_kwargs=plot_kwargs
        )

        # Panels c and f
        utils.lineplot(
            data,
            **data_eff_intb,
            **ticks_params_eff_intb[i],
            ax=axes[2, i + 1],
            plot_kwargs=plot_kwargs
        )

    # Range
    for i in range(3):
        axes[1, i].set_ylim(top=1.05)

    # Labels
    for i in range(3):
        axes[i, 1].set_ylabel("")
        axes[i, 2].set_ylabel("")
    for i in range(2):
        axes[0, i + 1].set_xlabel("")
        axes[1, i + 1].set_xlabel("")
        axes[2, i + 1].set_xlabel("")
        axes[0, i + 1].tick_params(labelleft=False)
        axes[1, i + 1].tick_params(labelleft=False)
        axes[2, i + 1].tick_params(labelleft=False)
    axes[2, 1].set_xlabel(r"$p_s$", fontsize=18)
    axes[2, 0].set_xlabel("")

    # Legend
    axes[0, 1].legend().remove()
    axes[0, 2].legend().remove()
    axes[0, 0].legend(frameon=False, loc="upper right", bbox_to_anchor=[1.0, 1.05])

    # Titles
    fig.text(0.24, 0.9, "Unipartite network", ha="center", va="center", fontsize=14)
    fig.text(
        0.64, 0.9, "People-Gathering network", ha="center", va="center", fontsize=14
    )

    axes[0, 1].annotate(
        "People:Gathering=3:1",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=12,
    )
    axes[0, 2].annotate(
        "People:Gathering=9:1",
        xy=(0.1, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=12,
    )

    # Subcaption
    for i in range(9):
        subcap = "cdefghijklmn"[i]
        xy = (0.05, 0.98)
        if subcap == "d":
            xy = (0.08, 0.9)
        if subcap == "d":
            xy = (0.08, 0.9)
        if subcap == "j":
            xy = (0.08, 0.82)

        c, r = divmod(i, 3)
        axes[r, c].annotate(
            subcap, xy=xy, xycoords="axes fraction", ha="left", va="top", fontsize=20,
        )

    fig.subplots_adjust(wspace=0.08)
    fig.savefig(FIG_FILE, bbox_inches="tight", dpi=300)
