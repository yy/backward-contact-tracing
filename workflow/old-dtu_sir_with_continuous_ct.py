import sys

import numpy as np

import pandas as pd
import utils_dtu


def eval_intervention(interv_func, tree_list, p_sample, params):

    num_prevented = np.zeros(len(tree_list)) * np.nan
    num_isolated = np.zeros(len(tree_list)) * np.nan
    num_new_cases = np.zeros(len(tree_list)) * np.nan
    for i, tree in enumerate(tree_list):

        isolated, interv_t = interv_func(tree, p_sample=p_sample, **params)

        preventable_cases = utils_dtu.get_preventable_cases(
            tree, interv_t, np.inf, isolated
        )

        new_cases = utils_dtu.find_cases(tree, params["start_t"], np.inf)

        num_isolated[i] = len(set(isolated))
        num_prevented[i] = len(set(preventable_cases))
        num_new_cases[i] = len(set(new_cases))

    df = pd.DataFrame(
        {
            "num_prevented": num_prevented,
            "num_isolated": num_isolated,
            "num_new_cases": num_new_cases,
            "p_sample": p_sample,
        }
    )
    return df


if __name__ == "__main__":

    # Input
    contact_data = sys.argv[1]
    sim_log_data = sys.argv[2]

    trace_time_window_day = float(sys.argv[3])  # time window for tracing contact
    close_contact_hour_per_day = float(sys.argv[4])
    p_t = float(sys.argv[5])  # prob. of conducting the contact tracing
    p_r = float(sys.argv[6])  # prob of isolating the contacts
    max_trace_nodes = int(sys.argv[7])  # number of nodes to isolate
    interv_start_day = float(
        sys.argv[8]
    )  # above which we carry out the contact tracing
    SAMPLE_NUM = int(sys.argv[9])

    # Output
    OUTPUT = sys.argv[10]

    # Data
    contact_logs = pd.read_csv(contact_data, sep=",")
    sim_logs = pd.read_csv(sim_log_data, sep=",")

    contact_list = utils_dtu.make_contact_list(sim_logs, contact_logs)

    tree_list = utils_dtu.construct_transmission_tree(sim_logs)

    tree_list = [tree_list[i] for i in range(SAMPLE_NUM)]

    # Set the parameters
    interv_params = {
        "contact_list": contact_list,
        "start_t": interv_start_day * 12 * 24,
        "p_t": p_t,  # Probability of conducting contact tracing
        "p_r": p_r,  # Probability that a traced node is isolated
        "trace_time_window": trace_time_window_day
        * 24
        * 12,  # 24 * 12: transform day scale to 5 mins time scale
        "close_contact_threshold": trace_time_window_day
        * close_contact_hour_per_day
        * 12,
        "max_trace_nodes": max_trace_nodes,
        "intervention_cycle": 1,
    }

    result_table = []
    for p_sample in (
        np.linspace(0.01, 0.1, 10).tolist() + np.linspace(0.2, 1.0, 11).tolist()
    ):
        result_table += [
            eval_intervention(
                utils_dtu.continuous_contact_tracing,
                tree_list,
                p_sample=p_sample,
                params=interv_params,
            )
        ]

    result_table = pd.concat(result_table, ignore_index=True)
    result_table.to_csv(OUTPUT, sep="\t")
