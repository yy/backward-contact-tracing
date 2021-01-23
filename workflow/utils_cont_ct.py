import logging

import networkx as nx
import numpy as np

import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def construct_transmission_tree(sim_logs):
    tree_list = []

    sim_model = "sir"
    if np.any(sim_logs.type == "e"):
        sim_model = "seir"

    for sim_id, sim_log in sim_logs.groupby("id"):

        se_event = sim_log.dropna()
        # se_event = sim_log[sim_log["type"] == compartment].dropna()

        se_event["source"] = se_event["source"].astype(int)
        se_event["node"] = se_event["node"].astype(int)

        tree = nx.from_pandas_edgelist(
            df=se_event,
            source="source",
            target="node",
            edge_attr="elapsed",
            create_using=nx.DiGraph,
        )

        # Find the root of the tree
        first_cases = [n for n, d in tree.in_degree() if d == 0]
        infected_time = dict(zip(first_cases, np.zeros(len(first_cases))))
        nx.set_node_attributes(tree, infected_time, "infected_time")

        # Set the time of infection as an attribute of each node
        if sim_model == "seir":
            infected_time = sim_log[sim_log.type == "e"][["node", "elapsed"]]
            infected_time = dict(zip(infected_time.node, infected_time.elapsed))
            nx.set_node_attributes(tree, infected_time, "infected_time")

            # Set the time of onset as an attribute of each node
            infected_time = sim_log[sim_log.type == "i"][["node", "elapsed"]]
            infected_time = dict(zip(infected_time.node, infected_time.elapsed))
            nx.set_node_attributes(tree, infected_time, "onset_time")
        elif sim_model == "sir":
            infected_time = sim_log[sim_log.type == "i"][["node", "elapsed"]]
            infected_time = dict(zip(infected_time.node, infected_time.elapsed))
            nx.set_node_attributes(tree, infected_time, "infected_time")

        # Set the time of recovery as an attribute of each node
        recovery_time = sim_log[sim_log.type == "r"][["node", "elapsed"]]
        recovery_time = dict(zip(recovery_time.node, recovery_time.elapsed))
        nx.set_node_attributes(tree, recovery_time, "recovery_time")

        # Add three to the list
        tree_list += [tree]
    return tree_list


def make_contact_history(sim_logs, contact_logs, t_offset=0):
    max_sim_timestamp = sim_logs["elapsed"].max()
    max_contact_timestamp = contact_logs["timestamp"].max()
    num_cycles = np.ceil(max_sim_timestamp / max_contact_timestamp).astype(int)

    # Repeat the contact data num_cycles times
    contact_logs_list = []
    for i in range(num_cycles):
        _next_cycle = contact_logs.copy()
        _next_cycle["timestamp"] += max_contact_timestamp * i
        contact_logs_list += [_next_cycle]

    contact_logs = pd.concat(contact_logs_list, ignore_index=True)

    contact_logs["timestamp"] = contact_logs["timestamp"] - t_offset
    contact_logs = contact_logs[contact_logs["timestamp"] >= 0]

    # Make a contact list for each node
    contact_logs = pd.concat(
        [
            contact_logs.rename(columns={"n1": "node", "n2": "contact"}),
            contact_logs.rename(columns={"n2": "node", "n1": "contact"}),
        ],
        ignore_index=True,
    )
    contact_list = {
        i: contact
        for i, contact in contact_logs.sort_values(by="timestamp").groupby("node")
    }
    return contact_list


def do_case_isolation(tree, p_s, start_t, case_isolatable_nodes):
    """
    Return a dict, where keys are the nodes and values are the time of isolation
    """
    nodes = np.array(list(tree.nodes()))
    case_isolated = nodes[np.random.rand(len(nodes)) <= p_s]
    case_isolated = [x for x in case_isolated if tree.nodes[x]["onset_time"] >= start_t]

    # Remove nodes that avoid being infected by case isolation for other nodes
    prevented_cases = set([])
    for node in case_isolated:
        for child in tree.successors(node):
            if tree.nodes[child]["infected_time"] > tree.nodes[node]["onset_time"]:
                prevented_cases.update(nx.descendants(tree, child))
                prevented_cases.add(child)
    if case_isolatable_nodes is None:
        case_isolated = np.array(list(set(case_isolated).difference(prevented_cases)))
    else:
        case_isolated = set(case_isolated).difference(prevented_cases)
        case_isolated = case_isolated.intersection(case_isolatable_nodes)
        case_isolated = np.array(list(case_isolated))

    # Compute the time of isolation
    event_list = np.array([tree.nodes[node]["onset_time"] for node in case_isolated])

    # Undo the case isolation for nodes isolated before start_t
    # case_isolated = case_isolated[event_list >= start_t]
    # event_list = event_list[event_list >= start_t]

    return dict(zip(case_isolated, event_list))


def do_contact_trace(
    case_isolated_nodes,
    dt,
    memory_dt,
    get_contact_list,
    p_t,
    top,
    tree,
    trace_mode="frequency",
):
    """
    Contact tracing

    Params
    ------
    case_isolated_nodes : dict
    dt : float
        Interval of contact tracing
    get_contact_list : func
        get_contact_list(i, t) returns the list of
        recent close contact of node i at time t
    p_t : float
        Probability of conducting the contact tracing
    top : int
        Maximum number of nodes traced. The most frequen nodes will be
        chosen.
    tree : nx.DiGraph
        Transmission tree
    already_isolated: set
        Set of nodes that are already isolated. Such nodes will not be
        included in the traced nodes

    Return
    ------
        Set of nodes to be traced by the contact tracing
    """
    if top == 0:
        return {}

    # Get the nodes and the time of case isolation
    nodes = np.array(list(case_isolated_nodes.keys()))
    event_time = np.array(list(case_isolated_nodes.values()))
    already_isolated = set(nodes)

    # Sample nodes for contact tracing
    # sampled = np.random.rand(nodes.size) <= p_t
    # sampled_nodes = nodes[sampled]
    # sampled_event_time = event_time[sampled]

    # Grouping
    group_ids = np.ceil(event_time / dt)

    # Carry out the contact tracing for each group
    nodes_ctrace = {}
    prevented_cases = set([])

    if len(group_ids) > 0:
        gid_list = np.arange(np.min(group_ids), np.max(group_ids) + 1)
    else:
        gid_list = []
    traced_history = []
    for group_id in gid_list:
        time_of_ct = group_id * dt

        # Get the contact list and concatenate them
        starting_nodes = nodes[group_id == group_ids]
        starting_nodes = set(starting_nodes).difference(prevented_cases)

        traced_t = [get_contact_list(x, group_id * dt) for x in starting_nodes]

        # Concatenate the list of lists into a flat list
        traced_t = sum(traced_t, [])
        traced_t = np.array(traced_t)

        # Random samples
        if len(traced_t) > 0:
            traced_t = traced_t[np.random.rand(len(traced_t)) <= p_t]

        # Update traced list
        traced_history = [
            x
            for x in traced_history
            if (x["t"] >= (time_of_ct - memory_dt))
            and (x["node"] not in already_isolated)
        ]
        traced_history += [{"t": time_of_ct, "node": x} for x in traced_t]

        if len(traced_history) > 0:
            traced = [x["node"] for x in traced_history]
            if trace_mode == "frequency":
                traced, freq = np.unique(traced, return_counts=True)
                node_order = np.argsort(freq)[::-1]
            elif trace_mode == "random":
                traced = np.unique(traced)
                node_order = np.argsort(np.random.rand(traced.size))[::-1]

            traced_num = 0
            for i, order in enumerate(node_order):
                node_id = traced[order]
                # not case isolated yet and not contact traced yet
                if (node_id not in already_isolated) and (node_id not in nodes_ctrace):
                    nodes_ctrace[node_id] = time_of_ct
                    already_isolated.add(node_id)
                    if node_id in tree:
                        for child in tree.successors(node_id):
                            if tree.nodes[child]["infected_time"] > time_of_ct:
                                prevented_cases.update(nx.descendants(tree, child))
                    traced_num += 1
                    if traced_num == top:
                        break

    unprotected_case_isolated = set(nodes).difference(prevented_cases)
    unprotected_case_isolated = {
        node: case_isolated_nodes[node] for node in unprotected_case_isolated
    }
    return {**unprotected_case_isolated, **nodes_ctrace}


def get_unprevented_cases(roots, tree, parent_isolated_time, t_isolated):
    def _get_unprevented_cases(node, tree, parent_isolated_time, t_isolated):
        unprevented_cases = []
        isolated_time = t_isolated.get(node, np.inf)
        if (
            np.minimum(isolated_time, parent_isolated_time)
            < tree.nodes[node]["infected_time"]
        ):
            return unprevented_cases

        unprevented_cases += [int(node)]
        for child in tree.successors(node):
            unprevented_cases += _get_unprevented_cases(
                child, tree, isolated_time, t_isolated
            )
        return unprevented_cases

    if isinstance(roots, list):
        nodes = []
        for root in roots:
            nodes += _get_unprevented_cases(
                root, tree, parent_isolated_time, t_isolated
            )
        return nodes
    else:
        return _get_unprevented_cases(roots, tree, parent_isolated_time, t_isolated)


def find_cases(tree, t_min):
    return set(
        [int(x[0]) for x in tree.nodes(data=True) if t_min < x[1]["infected_time"]]
    )


def find_root_in_subgraph(tree, nodes):
    tree_cycle = tree.subgraph(nodes)
    return set([nn for nn, dd in tree_cycle.in_degree() if dd == 0])


def find_parent(node, trans_tree):
    if node in trans_tree and list(trans_tree.predecessors(node)):
        parent = list(trans_tree.predecessors(node))[0]
        return parent
    else:
        return None


def set_onset_time(tree, dt):
    onset_time = {x[0]: x[1]["infected_time"] + dt for x in tree.nodes(data=True)}
    nx.set_node_attributes(tree, onset_time, "onset_time")
    return tree


def get_infected_nodes_by_t(tree, unprevented_nodes):

    plus = np.array([tree.nodes[x]["infected_time"] for x in unprevented_nodes])
    minus = np.array(
        [tree.nodes[x].get("recovery_time", np.inf) for x in unprevented_nodes]
    )
    plus = np.vstack([np.ones(plus.size), plus]).T
    minus = np.vstack([-np.ones(minus.size), minus]).T

    event_list = np.vstack([plus, minus])
    order = np.argsort(event_list[:, 1])
    event_list = event_list[order, :]
    # event_list[:,1] = np.cumsum(event_list[:,1])
    return pd.DataFrame(event_list, columns=["I", "time"])


def simulate(
    tree_list,
    start_t,
    cycle_dt,
    memory_dt,
    get_contact_list,
    p_s,
    max_traced_nodes,
    trace_mode,
    case_isolatable_nodes=None,
):

    result_table = []
    event_table = []
    for tid, tree in enumerate(tree_list):

        # Tree-dependent stats
        all_cases = find_cases(
            tree, start_t
        )  # total number of infections after start_t
        num_all_cases = len(set(all_cases))
        tree_roots = [nn for nn, dd in tree.in_degree() if dd == 0]

        # No intervention
        _event_table = get_infected_nodes_by_t(
            tree, set([int(x[0]) for x in tree.nodes(data=True)])
        )
        _event_table["p_s"] = -1
        _event_table["tid"] = tid
        _event_table["intervention"] = "no intervention"
        event_table += [_event_table]

        for p_t in (
            [0]
            + np.linspace(0.01, 0.1, 10).tolist()
            + np.linspace(0.2, 1.0, 9).tolist()
        ):

            isolated_case_isolation = do_case_isolation(
                tree, p_s, start_t, case_isolatable_nodes
            )

            # Contact tracing
            #    Note: isolated_ctraced includes both the case isolated and trace
            #          isolated nodes
            isolated_ctraced = do_contact_trace(
                isolated_case_isolation,
                cycle_dt,
                memory_dt,
                get_contact_list,
                p_t,
                max_traced_nodes,
                tree,
                trace_mode,
            )

            # Get the prevented cases by the case isolation
            unprevented_case_isolation = get_unprevented_cases(
                tree_roots, tree, start_t, isolated_case_isolation
            )
            prevented_case_isolation = all_cases.difference(
                set(unprevented_case_isolation)
            )

            # Get the prevented cases by the case isolation + contact tracing
            unprevented_ctraced = get_unprevented_cases(
                tree_roots, tree, start_t, isolated_ctraced
            )
            prevented_ctraced = all_cases.difference(set(unprevented_ctraced))

            # Count
            num_isolated_case_isolation = len(isolated_case_isolation)
            num_prevented_case_isolation = len(prevented_case_isolation)
            num_isolated_ctraced = len(isolated_ctraced)
            num_prevented_ctraced = len(prevented_ctraced)

            # Get time vs number of infected nodes
            _event_table = get_infected_nodes_by_t(tree, unprevented_ctraced)
            _event_table["p_s"] = p_s
            _event_table["p_t"] = p_t
            _event_table["tid"] = tid
            _event_table["intervention"] = "intervention"
            event_table += [_event_table]

            # logging
            df = pd.DataFrame(
                [
                    {
                        "num_prevented": num_prevented_case_isolation,
                        "num_isolated": num_isolated_case_isolation,
                        "num_contact_traced": 0,
                        "num_all_cases": num_all_cases,
                        "intervention": "case isolation",
                        "p_sample": p_s,
                        "p_t": p_t,
                    },
                    {
                        "num_prevented": num_prevented_ctraced,
                        "num_isolated": num_isolated_ctraced,
                        "num_all_cases": num_all_cases,
                        "intervention": "contact tracing",
                        "p_sample": p_s,
                        "p_t": p_t,
                    },
                ]
            )
            logging.debug(
                "p_t={p_t:.2f}, p_s={p_s:.2f}, isolated: ci={ci_isolated}, "
                "ct={ct_isolated}, prevented: ci={ci_prevented}, ct={ct_prevented}, "
                "efficiency: ci={ci_efficiency:.2f}, ct={ct_efficiency:.2f}".format(
                    p_s=p_s,
                    p_t=p_t,
                    ci_isolated=num_isolated_case_isolation,
                    ct_isolated=num_isolated_ctraced,
                    ci_prevented=num_prevented_case_isolation,
                    ct_prevented=num_prevented_ctraced,
                    ci_efficiency=num_prevented_case_isolation
                    / np.maximum(1, num_isolated_case_isolation),
                    ct_efficiency=num_prevented_ctraced
                    / np.maximum(1, num_isolated_ctraced),
                )
            )
            result_table += [df]

    return (
        pd.concat(result_table, ignore_index=True),
        pd.concat(event_table, ignore_index=True),
    )
