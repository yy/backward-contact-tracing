import logging
import sys

import pandas as pd
import utils_cont_ct as utils

logging.basicConfig(level=logging.DEBUG)


def make_get_contact_list_func(contact_history, time_window, close_contact_threshold):
    """
    Parameters
    ----------
    contact_history:
        History of contacts. See utils.make_contact_history(sim_logs, contact_logs)
    time_window: int
        We will make a list of contacts made between (t-time_window,t]
    close_contact_threshold: int
        If a node has a contact with another node more than
        ``close_contact_threshold'', we consider the contact a
        a close contact. Other contacts will be dropped from the
        return. Set = 0 to retrieve all recent contacts.

    Returns
    -------
    func : func
        func(i, t) returns the contact list at time t for node i.
    """

    def get_recent_close_contacts(node_id, t):
        contacts = contact_history.get(node_id, None)

        if contacts is None:
            return []

        recent_contacts = contacts[
            ((t - time_window) < contacts["timestamp"]) * (contacts["timestamp"] <= t)
        ]
        freq = recent_contacts["contact"].value_counts()
        return [k for k, v in freq[freq > close_contact_threshold].items()]

    return get_recent_close_contacts


if __name__ == "__main__":

    #
    # Input
    #
    contact_data = sys.argv[1]
    sim_log_data = sys.argv[2]
    sim_meta_data = sys.argv[3]
    trace_time_window_day = float(sys.argv[4])  # time window for tracing contact
    close_contact_hour_per_day = float(sys.argv[5])
    p_s = float(sys.argv[6])  # detection probability
    max_traced_nodes = int(sys.argv[7])  # number of nodes to isolate
    start_t = float(sys.argv[8])  # above which we carry out the contact tracing
    cycle_dt = float(sys.argv[9])  # above which we carry out the contact tracing
    memory_dt = float(sys.argv[10])  # above which we carry out the contact tracing
    time_lag_for_isolation = float(sys.argv[11])
    trace_mode = sys.argv[12]
    OUTPUT = sys.argv[13]
    OUTPUT_EVENT = sys.argv[14]

    #
    # Recale the time scale
    #
    start_t = start_t * 12 * 24
    trace_time_window = (
        trace_time_window_day * 24 * 12
    )  # 24 * 12: transform day scale to 5 mins time scale
    memory_dt = memory_dt * 12 * 24
    close_contact_threshold = trace_time_window_day * close_contact_hour_per_day * 12
    time_lag_for_isolation = time_lag_for_isolation * 12 * 24

    #
    # Load data
    #
    logging.debug("Load data")
    contact_logs = pd.read_csv(contact_data, sep=",")
    sim_logs = pd.read_csv(sim_log_data, sep=",")
    sim_meta_data = pd.read_csv(sim_meta_data, sep=",")

    #
    # Preprocess
    #
    logging.debug("Extract off set time")
    t_offset = dict(zip(sim_meta_data["id"].values, sim_meta_data["t"].values))

    #
    # Simulation
    #
    result_table = []
    event_table = []
    tid = 0
    for sim_id, log_file in sim_logs.groupby("id"):

        """
        Note:
        sim_log timestamp is an elapse time, time since the first infection.
        On the other hand, the timestamp in the contact data is the actual time.
        Thus, the alignment is necessary. Here, I convert the timestamp to the
        elapse time by offsetting.
        """
        t_0 = t_offset.get(sim_id)

        logging.debug("Make contact list")
        contact_history = utils.make_contact_history(
            log_file, contact_logs, t_offset=t_0
        )

        logging.debug("Make transmission tree")
        tree = utils.construct_transmission_tree(log_file)

        logging.debug("Generate the contact list func")
        get_contact_list = make_get_contact_list_func(
            contact_history, trace_time_window, close_contact_threshold
        )

        rt, et = utils.simulate(
            tree,
            start_t,
            cycle_dt,
            memory_dt,
            get_contact_list,
            p_s,
            max_traced_nodes,
            trace_mode,
        )
        et["tid"] = tid
        result_table += [rt]
        event_table += [et]
        tid += 1
    result_table = pd.concat(result_table, ignore_index=True)
    event_table = pd.concat(event_table, ignore_index=True)
    result_table.to_csv(OUTPUT, sep="\t")
    event_table.to_csv(OUTPUT_EVENT, sep="\t")
