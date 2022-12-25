import utils
import numpy as np
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt


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


def generate_perturb_df(best_distance, best_perturb, feat_columns):
    df_dist = pd.DataFrame({"id": range(len(best_distance)), "best_distance": best_distance})
    df_perturb = pd.DataFrame(best_perturb, columns=feat_columns[:-1])
    return pd.concat([df_dist, df_perturb], axis=1)


def generate_perturbed_df(df_perturb, feat_input):
    return pd.DataFrame(df_perturb + feat_input)


def plot_pertubed(df_perturb: pd.DataFrame):
    f, ax = plt.subplots(figsize=(3, 7))
    sns.barplot(x=np.mean(df_perturb.iloc[:, 2:], axis=0), y=df_perturb.iloc[:, 2:].columns)
    ax.set(xlabel="Average pertubation per column")
