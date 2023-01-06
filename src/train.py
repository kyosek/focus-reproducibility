import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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
            n_estimators=100, max_depth=2, random_state=42
        )
    elif model_type == "ab":
        # dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        # model = AdaBoostClassifier(
        #     base_estimator=dt,
        #     n_estimators=n_estimators,
        #     learning_rate=lr,
        #     random_state=42,
        # )
        model = AdaBoostClassifier(
            n_estimators=100,
            random_state=42,
        )

    model.fit(x_train, y_train)
    pickle.dump(model, open("my_models/" + model_type + "_" + data_name + ".pkl", "wb"))
    print("train completed")


if __name__ == "__main__":
    train_model(
        model_type="ab",
        data_name="cf_heloc_data_train",
        max_depth=4,
    )
