import os
import re
import sys

import numpy as np

import pandas as pd
from tqdm import tqdm


def load_data(filename_list):
    def get_parameters(filename):
        """
        Get parameter values from the filename
        """
        filename = filename.replace(".gz", "")
        filename = os.path.splitext(os.path.basename(filename))[0]
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

    def preprocess(result_table):
        # Drop results with no new case
        result_table = result_table[result_table["num_all_cases"] > 0]
        result_table["f_num_prevented"] = (result_table["num_prevented"]) / np.maximum(
            1.0, result_table["num_all_cases"]
        )

        result_table["efficiency"] = result_table["num_prevented"] / np.maximum(
            result_table["num_isolated"], 1
        )

        result_table = result_table.rename(
            columns={
                "p_sample": "p_s",
                "num_prevented": "Preventable cases",
                # "f_num_prevented": "Preventable cases",
                "num_isolated": "Isolated cases",
            }
        )
        return result_table

    tables = []
    for filename in tqdm(filename_list):
        df = pd.read_csv(filename, sep="\t")
        df.loc[pd.isna(df["p_t"]), "p_t"] = 0
        # df.loc[pd.isna(df["intervention"]), "intervention"] = "intervention"
        df = preprocess(df)
        params = get_parameters(filename)
        for k, v in params.items():
            df[k] = v
        # df["intervention"] = interv
        # df["p_s"] = p_s
        tables += [df]
    table = pd.concat(tables, ignore_index=True)
    return table


if __name__ == "__main__":

    OUTPUT = sys.argv.pop()
    FILE_LIST = sys.argv[1]
    INPUT_FILES = sum(pd.read_csv(FILE_LIST).values.tolist(), [])
    df = load_data(INPUT_FILES)
    df.to_csv(OUTPUT, sep="\t", index=False)
