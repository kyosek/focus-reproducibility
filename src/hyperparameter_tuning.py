import numpy as np
import pandas as pd
import optuna
from src.approximation import compute_cfe
import pickle


def objective(trial):
    model_algo = "dt"
    opt = "adam"
    num_iter = 1000
    distance_function = "mahal"
    # "mahal"cosine"euclidean"l1

    data_name = "cf_compas_num_data_test"

    try:
        unchanged_df = pd.read_csv(f"{data_name}_unchanged.csv")
    except FileNotFoundError:
        unchanged_df = []

    df = pd.read_csv("data/{}.tsv".format(data_name), sep="\t", index_col=0)
    feat_matrix = df.values.astype(float)

    feat_input = feat_matrix[:, :-1]

    train_name = data_name.replace("test", "train")
    train_data = pd.read_csv("data/{}.tsv".format(train_name), sep="\t", index_col=0)
    x_train = np.array(train_data.iloc[:, :-1])

    # had to match the scikit-learn version to 0.21.3 in order to load the model but eventually upgrade it
    # model = joblib.load("models/{}".format(model_name), "rb")
    model = pickle.load(
        open("my_models/" + model_algo + "_" + train_name + ".pkl", "rb")
    )

    # add if model_type == "DT", temperature_val = 0
    unchanged_ever, cfe_distance, best_perturb = compute_cfe(
        model,
        feat_input,
        distance_function,
        opt,
        sigma_val=trial.suggest_int("sigma", 1, 20),
        # temperature_val=trial.suggest_int("temperature", 1, 20),
        temperature_val=0,
        distance_weight_val=round(
            trial.suggest_float("distance_weight", 0.01, 0.1, step=0.01), 2
        ),
        lr=round(trial.suggest_float("lr", 0.001, 0.01, step=0.001), 3),
        # lr=0.003,
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
        f"{data_name}_unchanged.csv"
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
