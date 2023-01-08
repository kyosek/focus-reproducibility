import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cfe_dist_box(savefig=False):
    data_list = ["compas", "heloc", "shop2", "wine"]
    for data in data_list:
        original_df = pd.read_csv(f"visualisation_data/original_cfe_dt_{data}.csv")
        original_df.columns = ["model", "cfe_distance"]
        original_df["model"] = "Original"

        new_df = pd.read_csv(f"visualisation_data/new_cfe_dt_{data}.csv")
        new_df.columns = ["model", "cfe_distance"]
        new_df["model"] = "New"

        df = pd.concat([original_df, new_df])

        sns.boxplot(data=df, x="model", y="cfe_distance")
        plt.title(f"CE distance distribution of original and new model")

        if savefig:
            plt.savefig(f"plots/{data}_ce_dist.png")
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    plot_cfe_dist_box(True)
