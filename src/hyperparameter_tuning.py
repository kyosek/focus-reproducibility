import numpy as np
import pandas as pd
import optuna
from src.counterfactual_explanation import compute_cfe
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("model_type", type=str)
parser.add_argument("num_iter", type=int, default=100)
parser.add_argument("sigma", type=float, default=1.0)
parser.add_argument("temperature", type=float, default=1.0)
parser.add_argument("distance_weight", type=float, default=0.01)
parser.add_argument("lr", type=float, default=0.001)
parser.add_argument(
    "opt",
    type=str,
    default="adam",
    help="Options are either adam or gd (as str)",
)
parser.add_argument("data_name", type=str)
parser.add_argument("distance_function", type=str)
parser.add_argument("n_trials", type=int)

args = parser.parse_args()
model_type = args.model_type
num_iter = args.num_iter
sigma_val = args.sigma
distance_weight_val = args.distance_weight
lr = args.lr
opt = args.opt
data_name = args.data_name
distance_function = args.distance_function


def objective(trial):
    """
    This function is an objective function for hyperparameter tuning using optuna.
    It trains and evaluates a decision tree model for a given dataset and computes the Counterfactual Explanation (CFE)
    distance and unchanged rate for a set of hyperparameters.

    Args:
    trial (optuna.Trial): Object that contains information about the current trial, including hyperparameters.

    Returns:
    CFE distance mean and unchanged rate squared, as an objective function for hyperparameter optimization.

    * note: typically we want to minimise a number of unchanged first, so penalising the score by having squared number.
    Also, to not distort this objective, having the mean distance divided by 100.
    """
    try:
        unchanged_df = pd.read_csv(f"visualisation_data/{data_name}_unchanged.csv")
    except FileNotFoundError:
        unchanged_df = []

    df = pd.read_csv("data/{}.tsv".format(data_name), sep="\t", index_col=0)
    feat_matrix = df.values.astype(float)

    feat_input = feat_matrix[:, :-1]

    train_name = data_name.replace("test", "train")
    train_data = pd.read_csv("data/{}.tsv".format(train_name), sep="\t", index_col=0)
    x_train = np.array(train_data.iloc[:, :-1])

    model = pickle.load(
        open("retrained_models/" + model_type + "_" + train_name + ".pkl", "rb")
    )

    # DT models do not use temperature
    if model_type == "dt":
        temperature_val = 0
    else:
        temperature_val = trial.suggest_int("temperature", 1, 20, step=1.0)

    unchanged_ever, cfe_distance, best_perturb = compute_cfe(
        model,
        feat_input,
        distance_function,
        opt,
        sigma_val=trial.suggest_int("sigma", 1, 20, step=1.0),
        temperature_val=temperature_val,
        distance_weight_val=round(
            trial.suggest_float("distance_weight", 0.01, 0.1, step=0.01), 2
        ),
        lr=round(trial.suggest_float("lr", 0.001, 0.01, step=0.001), 3),
        num_iter=num_iter,
        x_train=x_train,
        verbose=0,
    )

    if type(unchanged_df) == list:
        unchanged_df.append(unchanged_ever)
    else:
        unchanged_df = unchanged_df.append(
            {"unchanged": unchanged_ever}, ignore_index=True
        )
    pd.DataFrame(unchanged_df, columns=["unchanged"]).to_csv(
        f"visualisation_data/{data_name}_unchanged.csv"
    )

    print(f"Unchanged: {unchanged_ever}")
    print(f"Mean distance: {np.mean(cfe_distance)}")

    return np.mean(cfe_distance) / 100 + pow(unchanged_ever, 2)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")

    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(optuna.importance.get_param_importances(study))
