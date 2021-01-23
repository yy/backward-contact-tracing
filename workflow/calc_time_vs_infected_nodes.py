import os
import re
import sys

import numpy as np
from scipy import sparse

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def load_data(filename_list, num_time_points):
    def get_parameters(filename):
        """
        Get parameter values from the filename
        """
        filename = filename.replace(".gz", "")
        filename = os.path.splitext(os.path.basename(filename))[0]
        print(filename)
        filename = filename.replace("-", "|").replace("_", "|")
        res = {}
        for r in filename.split("|"):
            match = re.match(r"([a-z,A-Z]+)[=]*([0-9,.,a-z,A-Z]+)", r, re.I)
            if match:
                items = match.groups()
                k = items[0]
                v = items[1]
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                res[k] = v
        return res

    def table2It(df, time_points):
        """
        Read the given event table with columns I and time.
        Then, count the number of infected nodes at each time.
        """
        # Construct a matrix C with entry C[i,t] means
        # number of infected individuals at time t
        df = df.sort_values(by="time")
        tree_id, timestamp, infected = (
            df["tid"].values,
            df["time"].values,
            df["I"].values,
        )
        recovered = -infected.copy()
        recovered[recovered < 0] = 0

        # digitize the time
        timestamp, time_id = np.unique(timestamp, return_inverse=True)

        # Compute the number of infected at each event of time
        tree_id = tree_id.astype(int)
        It = sparse.csr_matrix(
            (infected, (tree_id, time_id)),
            shape=(np.max(tree_id) + 1, np.max(time_id) + 1),
        ).toarray()
        Rt = sparse.csr_matrix(
            (recovered, (tree_id, time_id)),
            shape=(np.max(tree_id) + 1, np.max(time_id) + 1),
        ).toarray()

        order = np.argsort(timestamp)
        timestamp = timestamp[order]
        It = It[:, order]
        It = np.cumsum(It, axis=1)
        Rt = Rt[:, order]
        Rt = np.cumsum(Rt, axis=1)

        # Get the infected at each prescribed time point
        time_id = np.digitize(time_points, timestamp) - 1
        infected = It[:, time_id].reshape(-1)
        recovered = Rt[:, time_id].reshape(-1)
        timestamps = np.tile(time_points, It.shape[0])
        df = pd.DataFrame(
            {
                "time": timestamps,
                "Infected nodes": infected,
                "Recovered nodes": recovered,
            }
        )
        return df

    tmin = np.inf
    tmax = -np.inf
    print("Reading")
    n_jobs = 30

    def get_tmin_tmax(filename):
        df = pd.read_csv(filename, sep="\t")
        return df["time"].min(), df["time"].max()

    tmin_max_list = Parallel(n_jobs=n_jobs)(
        delayed(get_tmin_tmax)(filename) for filename in tqdm(filename_list)
    )

    tmin = np.min(np.array([x[0] for x in tmin_max_list]))
    tmax = np.max(np.array([x[1] for x in tmin_max_list]))

    def _table2It(filename, time_points):
        df = pd.read_csv(filename, sep="\t")
        params = get_parameters(filename)
        tables = []
        df.loc[pd.isna(df["p_t"]), "p_t"] = 0
        for interv, df_ in df.groupby("intervention"):
            for (p_s, p_t), dg in df_.groupby(["p_s", "p_t"]):
                dg = table2It(dg, time_points)
                dg["intervention"] = interv
                dg["p_s"] = p_s
                dg["p_t"] = p_t
                for k, v in params.items():
                    dg[k] = v
                tables += [dg]
        return pd.concat(tables, ignore_index=True)

    tables = []
    time_points = np.linspace(tmin, tmax, num_time_points)
    print("calc infected")
    tables = Parallel(n_jobs=n_jobs)(
        delayed(_table2It)(filename, time_points) for filename in tqdm(filename_list)
    )
    table = pd.concat(tables, ignore_index=True)
    return table


if __name__ == "__main__":

    OUTPUT = sys.argv.pop()
    NUM_TIME_POINTS = int(sys.argv.pop())
    FILE_LIST = sys.argv[1]
    INPUT_FILES = sum(pd.read_csv(FILE_LIST).values.tolist(), [])

    df = load_data(INPUT_FILES, NUM_TIME_POINTS)

    df.to_csv(OUTPUT, sep="\t", index=False)
