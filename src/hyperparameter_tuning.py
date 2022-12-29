import numpy as np
import pandas as pd
import optuna
from approximation import compute_cfe
import time
import pickle


def objective(trial):
    model_algo = "dt"
    opt = "adam"
    num_iter = 1000
    distance_function = "mahal"
    # "mahal"cosine"euclidean"l1

    data_name = "cf_shop2_data_test"

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

    unchanged_ever, counterfactual_examples, best_distance, best_perturb = compute_cfe(
        model,
        feat_input,
        distance_function,
        opt,
        sigma_val=trial.suggest_int("sigma", 1, 20),
        temperature_val=trial.suggest_int("temperature", 1, 20),
        distance_weight_val=round(trial.suggest_float("distance_weight", 0.01, 0.1, step=0.01), 2),
        lr=round(trial.suggest_float("learning_rate", 0.001, 0.1, step=0.001), 3),
        num_iter=num_iter,
        x_train=x_train,
        verbose=0,
    )

    print(f"Unchanged: {len(unchanged_ever)}")
    print(f"Mean distance: {np.mean(counterfactual_examples)}")

    return np.mean(counterfactual_examples) * (len(unchanged_ever) ** 2)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")

    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Multi-objectives
    # trial = study.best_trials
    # fig = optuna.visualization.plot_pareto_front(study)
    # fig.show()
