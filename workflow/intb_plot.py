import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

import pandas as pd
import seaborn as sns
import utils

rc("text", usetex=True)

FIG_FILE = sys.argv.pop()
RESULT_FILES = sys.argv[1:]

if __name__ == "__main__":

    p_r = 0.5  # prob. for conducting contact tracing
    frac_gathr_list = [0.25, 0.1]  # fraction of gathering nodes

    def load_data(result_files):
        """
        Data loader
        """
        res_df_list = []
        for resfile in result_files:
            with open(resfile, "rb") as f:
                result_param = json.load(f)  # load the data
            # Convert to the pandas DataFrame
            res_df = pd.DataFrame(
                {k: v for k, v in result_param.items() if k != "params"}
            )
            # Add some simulation parameters to the table
            params = result_param["params"]
            for added_param_name in params.keys():
                res_df[added_param_name] = params[added_param_name]
            # Append the table to the list
            res_df_list += [res_df]
        # Concatenate the tables into one big table
        result_table = pd.concat(res_df_list, ignore_index=True)

        return result_table

    def calculate_stats(df, n_ratio=1):
        """
        Stat calculator
        """
        numers = [
            "num_indirect_prev_cases",
            "num_direct_prev_cases",
            "num_prev_cases",
            "num_isolated",
        ]
        denom = ["num_new_cases", "num_new_cases", "num_new_cases", "num_nodes"]
        for j, numer in enumerate(numers):
            if denom[j] == "num_nodes":
                df["f_%s" % numer] = df[numer] / (df[denom[j]] * n_ratio)
            else:
                df["f_%s" % numer] = df[numer] / df[denom[j]]

        # Compute the efficacy
        df["protected_per_isolated"] = df["num_prev_cases"] / df["num_isolated"]
        return df

    def split_with_without_contact_tracing(df, p_r):
        return df[df.p_r == p_r], df[df.p_r == 0]

    # Load data
    data4plot = []
    df_all = load_data(RESULT_FILES)
    df_all = calculate_stats(df_all)
    for frac_gathr in frac_gathr_list:
        df = df_all[df_all.frac_gathr == frac_gathr]
        df_with, df_without = split_with_without_contact_tracing(df, p_r)
        data4plot += [[df, df_without]]

    #
    # Set parameters for plot
    #
    sns.set(font_scale=1)
    sns.set_style("white")
    sns.set_style("ticks")

    cmap = sns.color_palette()
    plot_kwargs = {"ci": 95, "err_style": "band"}
    # Panels a and d
    data_isolated_cases = {
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
    ticks_params_isolated_cases = [
        {
            "xticks": np.linspace(0, 1, 6),
            "yticks": np.linspace(0, 0.05, 6),
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
    data_phi = {
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
    ticks_params_phi = [
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
    data_eff = {
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
    ticks_params_eff = [
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
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7.0, 6.5), sharex=True)

    for i, data in enumerate(data4plot):
        # Panels a and d
        utils.lineplot(
            data,
            **data_isolated_cases,
            **ticks_params_isolated_cases[i],
            ax=axes[0, i],
            plot_kwargs=plot_kwargs
        )

        # Panels b and e
        utils.lineplot(
            data,
            **data_phi,
            **ticks_params_phi[i],
            ax=axes[1, i],
            plot_kwargs=plot_kwargs
        )

        # Panels c and f
        utils.lineplot(
            data,
            **data_eff,
            **ticks_params_eff[i],
            ax=axes[2, i],
            plot_kwargs=plot_kwargs
        )

    # Labels
    for i in range(3):
        axes[i, 1].set_ylabel("")
    for i in range(2):
        axes[0, i].set_xlabel("")
        axes[1, i].set_xlabel("")
        axes[2, i].set_xlabel("")
    # Legend
    axes[0, 1].legend().remove()
    axes[0, 0].legend(frameon=False, loc="upper right", bbox_to_anchor=[1.0, 1.05])

    plt.subplots_adjust()

    # Shared x-axis
    fig.text(0.3, 0.04, r"$p_s$", fontsize=20)
    fig.text(0.72, 0.04, r"$p_s$", fontsize=20)
    for i in range(6):
        axes.flat[i].annotate(
            "fghijklmn"[i],
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=20,
        )
    fig.savefig(FIG_FILE, bbox_inches="tight", dpi=300)
