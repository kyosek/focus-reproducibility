import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def _evaluate_model(model, data_name: str):
    test_name = data_name.replace("train", "test")
    df = pd.read_csv("data/{}.tsv".format(test_name), sep="\t", index_col=0)
    feat_matrix = df.values.astype(float)

    x_test = feat_matrix[:, :-1]
    y_test = np.where(feat_matrix[:, -1] == -1, 0, 1)

    preds = model.predict(x_test)

    print("Prediction distribution:")
    print(pd.DataFrame(preds).describe())
    print("Accuracy score is:")
    print(accuracy_score(y_test, preds))


def train_model(
    model_type: str, data_name: str, max_depth: int, n_estimators=None, lr=None
):
    df = pd.read_csv("data/{}.tsv".format(data_name), sep="\t", index_col=0)
    feat_matrix = df.values.astype(float)

    x_train = feat_matrix[:, :-1]
    y_train = np.where(feat_matrix[:, -1] == -1, 0, 1)

    if model_type == "dt":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_type == "rf":
        # model = RandomForestClassifier(
        #     n_estimators=n_estimators, max_depth=max_depth, random_state=42
        # )
        model = RandomForestClassifier(
            n_estimators=100, max_depth=max_depth, random_state=42
        )
    elif model_type == "ab":
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model = AdaBoostClassifier(
            base_estimator=dt,
            # n_estimators=n_estimators,
            # learning_rate=lr,
            random_state=42,
        )
        # model = AdaBoostClassifier(
        #     n_estimators=500,
        #     random_state=42,
        # )

    model.fit(x_train, y_train)
    pickle.dump(
        model, open("retrained_models/" + model_type + "_" + data_name + ".pkl", "wb")
    )
    print("train completed")

    _evaluate_model(model, data_name)


if __name__ == "__main__":
    train_model(
        model_type="ab",
        data_name="cf_german_train",
        max_depth=2,
        n_estimators=100,
    )
