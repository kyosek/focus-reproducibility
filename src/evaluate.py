import utils
import numpy as np
import pandas as pd
import json


def generate_cf_stats(
    output_root: str,
    data_name: str,
    distance_function,
    unchanged_ever,
    counterfactual_examples,
):
    cf_stats = {
        "dataset": data_name,
        "distance_function": distance_function,
        "unchanged_ever": len(unchanged_ever),
        "mean_dist": np.mean(counterfactual_examples),
    }

    print("saving the text file")
    with utils.safe_open(output_root + "_cf_stats.txt", "w") as gsout:
        json.dump(cf_stats, gsout)


def generate_perturbed_df(n_examples, best_distance, best_perturb, feat_columns):
    df_dist = pd.DataFrame({"id": range(n_examples), "best_distance": best_distance})
    df_perturb = pd.DataFrame(best_perturb, columns=feat_columns[:-1])
    return pd.concat([df_dist, df_perturb], axis=1)


def generate_perturbed_df_diff(df_perturb, feat_input):
    return pd.DataFrame(df_perturb - feat_input)
