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

    def calculate_stats(df):
        """
        Stat calculator
        """
        # Normalize the cases
        numers = [
            "num_indirect_prev_cases",
            "num_direct_prev_cases",
            "num_prev_cases",
            "num_isolated",
        ]
        denom = ["num_new_cases", "num_new_cases", "num_new_cases", "num_nodes"]
        for j, numer in enumerate(numers):
            df["f_%s" % numer] = df[numer] / df[denom[j]]

        # Compute the efficacy
        df["protected_per_isolated"] = df["num_prev_cases"] / df["num_isolated"]

        # Compute the degree heterogeneity (expected value)
        df["gamma"] = 3 + df["price_c"] / df["price_m"]

        return df

    def split_with_without_contact_tracing(df, p_r):
        return df[df.p_r == p_r], df[df.p_r == 0]

    # Load data
    res_table = load_data(RESULT_FILES)
    res_table = calculate_stats(res_table)

    # Parameter config for Barabasi albert model
    df = res_table[(res_table.price_c == 0)]

    # Intervention with case isolation
    df_with, df_without = split_with_without_contact_tracing(df, p_r)
    data4plot = [df_with, df_without]

    #
    # Set parameters for plot
    #
    sns.set_style("white")
    sns.set(font_scale=1)
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
            "yticks": np.linspace(0, 0.1, 6),
            "inset_bbox": None,
            # "inset_bbox": [0.12, 0.47, 0.32, 0.32],
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
            "inset_bbox": [0.5, 0.2, 0.43, 0.43],
            "inset_xticks": np.linspace(0, 0.1, 3),
            "inset_yticks": np.linspace(0, 1, 3),
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
            "inset_bbox": [0.45, 0.42, 0.5, 0.5],
            "inset_xticks": np.array([0.01, 0.05, 0.1]),
            "inset_yticks": np.linspace(0, 15, 4),
        },
    ]

    #
    # Plot the figure
    #
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.5, 6.5), sharex=True)
    # fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 13))

    # Panels a and d
    utils.lineplot(
        data4plot,
        **data_isolated_cases,
        **ticks_params_isolated_cases[0],
        ax=axes[0],
        plot_kwargs=plot_kwargs
    )

    # Panels b and e
    utils.lineplot(
        data4plot,
        **data_phi,
        **ticks_params_phi[0],
        ax=axes[1],
        plot_kwargs=plot_kwargs
    )

    # Panels c and f
    utils.lineplot(
        data4plot,
        **data_eff,
        **ticks_params_eff[0],
        ax=axes[2],
        plot_kwargs=plot_kwargs
    )

    # Labels
    for i in range(2):
        axes[0].set_xlabel("")
        axes[1].set_xlabel("")
        axes[2].set_xlabel("")
    # Legend
    axes[0].legend(frameon=False, loc="upper right", bbox_to_anchor=[0.95, 1.05])

    plt.subplots_adjust()

    # Shared x-axis
    fig.text(0.5, 0.03, r"$p_s$", fontsize=20)
    for i in range(3):
        xy = (0.05, 0.98)
        if i == 1:
            xy = (0.05, 0.9)

        axes.flat[i].annotate(
            "cde"[i], xy=xy, xycoords="axes fraction", ha="left", va="top", fontsize=25,
        )
    fig.savefig(FIG_FILE, bbox_inches="tight", dpi=300)
