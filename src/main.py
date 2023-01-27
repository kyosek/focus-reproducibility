import numpy as np
from src.evaluate import (
    generate_perturb_df,
    generate_perturbed_df,
    generate_cf_stats,
    plot_perturbed,
)
from src.approximation import compute_cfe
import time
import pickle
import pandas as pd


# parser = argparse.ArgumentParser()
# parser.add_argument("--sigma", type=float, required=False, default=1.0)
# parser.add_argument("--temperature", type=float, required=False, default=1.0)
# parser.add_argument("--distance_weight", type=float, required=False, default=0.01)
# parser.add_argument("--lr", type=float, required=False, default=0.001)
# parser.add_argument(
#     "--opt",
#     type=str,
#     required=False,
#     default="adam",
#     help="Options are either adam or gd (as str)",
# )
# parser.add_argument("--model_name", type=str, required=True)
# parser.add_argument("--data_name", type=str, required=True)
# parser.add_argument("--model_type", type=str, required=True)
# parser.add_argument("--distance_function", type=str, required=True)

# args = parser.parse_args()
# sigma_val = args.sigma
# temperature_val = args.temperature
# distance_weight_val = args.distance_weight
# lr = args.lr
# opt = args.opt
# model_name = args.model_name
# data_name = args.data_name
# model_type = args.model_type
# distance_function = args.distance_function


def main():
    model_type = "dt"
    sigma_val = 7.0
    temperature_val = 3.0
    distance_weight_val = 0.01
    lr = 0.001
    opt = "adam"
    num_iter = 1000
    data_name = "cf_german_test"
    distance_function = "euclidean"
    # "mahal"cosine"euclidean"l1

    output_root = "hyperparameter_tuning/{}/{}/{}/perturbs_{}_sigma{}_temp{}_dweight{}_lr{}".format(
        distance_function,
        data_name,
        model_type,
        opt,
        sigma_val,
        temperature_val,
        distance_weight_val,
        lr,
    )

    start_time = time.time()

    df = pd.read_csv("data/{}.tsv".format(data_name), sep="\t", index_col=0)
    feat_columns = df.columns
    feat_matrix = df.values.astype(float)

    feat_input = feat_matrix[:, :-1]

    train_name = data_name.replace("test", "train")
    train_data = pd.read_csv("data/{}.tsv".format(train_name), sep="\t", index_col=0)
    x_train = np.array(train_data.iloc[:, :-1])

    model = pickle.load(
        open("my_models/" + model_type + "_" + train_name + ".pkl", "rb")
    )

    unchanged_ever, cfe_distance, best_perturb = compute_cfe(
        model,
        feat_input,
        distance_function,
        opt,
        sigma_val,
        temperature_val,
        distance_weight_val,
        lr,
        num_iter=num_iter,
        x_train=x_train,
        verbose=1,
    )

    # Evaluation
    generate_cf_stats(
        output_root,
        data_name,
        distance_function,
        unchanged_ever,
        cfe_distance,
        start_time,
    )

    pd.DataFrame(cfe_distance).to_csv(f"cfe_{model_type}_{train_name}.csv")

    # df_perturb = generate_perturb_df(best_distance, best_perturb, feat_columns)
    # df = generate_perturbed_df(best_perturb, feat_input)
    # plot_pertubed(df_perturb)


if __name__ == "__main__":
    main()
