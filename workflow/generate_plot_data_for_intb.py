import json
import sys

import pandas as pd

OUTPUT = sys.argv.pop()
RESULT_FILES = sys.argv[1:]

if __name__ == "__main__":

    # -----------------------
    # Load the result for int
    # -----------------------
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

    # Load data
    res_table = load_data(RESULT_FILES)
    df = calculate_stats(res_table)
    df.to_csv(OUTPUT, sep="\t", index=False)
