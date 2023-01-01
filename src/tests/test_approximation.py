import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from src.utils import filter_hinge_loss, calculate_distance
from src.approximation import _parse_class_tree, get_prob_classification_tree, get_prob_classification_forest, compute_cfe

dt_model = pickle.load(
        open("my_models/dt_cf_compas_num_data_train.pkl", "rb")
    )
rf_model = pickle.load(
        open("my_models/rf_cf_compas_num_data_train.pkl", "rb")
    )
df = pd.read_csv("data/cf_compas_num_data_test.tsv", sep="\t", index_col=0)
feat_input = df.values.astype(float)[:, :-1]
sigma = 5.0


def test__parse_class_tree():
    dt_leaf_nodes = _parse_class_tree(dt_model, feat_input, sigma)

    assert len(dt_leaf_nodes) == 2
    assert len(dt_leaf_nodes[0][0]) == 1852
    assert len(dt_leaf_nodes[1][0]) == 1852


def test_get_prob_classification_tree():
    pass


def test_get_prob_classification_forest():
    pass


def test_compute_cfe():
    pass
