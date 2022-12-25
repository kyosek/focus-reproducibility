import pickle
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def train_model(model_type: str, data_name: str, max_depth: int, n_estimator=None, lr=None):
    df = pd.read_csv("data/{}.tsv".format(data_name), sep="\t", index_col=0)
    feat_matrix = df.values.astype(float)

    # Remove the last column which is the label
    x_train = feat_matrix[:, :-1]
    y_train = feat_matrix[:, -1]

    if model_type == "dt":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimator=n_estimator, max_depth=max_depth, random_state=42)
    elif model_type == "ad":
        model = AdaBoostClassifier(n_estimator=n_estimator, max_depth=max_depth, learning_rate=lr, random_state=42)

    model.fit(x_train, y_train)
    pickle.dump(model, open("my_models/"+model_type+"_"+data_name+".pkl", 'wb'))
    print("train completed")


if __name__ == "__main__":
    train_model(model_type="dt", data_name="cf_compas_num_data_train", max_depth=4,)
