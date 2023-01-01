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
ab_model = pickle.load(
        open("my_models/ab_cf_compas_num_data_train.pkl", "rb")
    )
df = pd.read_csv("data/cf_compas_num_data_test.tsv", sep="\t", index_col=0)
feat_input = df.values.astype(float)[:, :-1]
sigma = 5.0
temperature = 10.0


def test__parse_class_tree():
    leaf_nodes = _parse_class_tree(dt_model, feat_input, sigma)

    assert len(leaf_nodes) == 2
    assert len(leaf_nodes[0][0]) == 1852
    assert len(leaf_nodes[1][0]) == 1852


def test_get_prob_classification_tree():
    dt_prob_list = get_prob_classification_tree(dt_model, feat_input, sigma)

    assert dt_prob_list.shape == (1852, 2)


def test_get_prob_classification_forest():
    rf_softmax = get_prob_classification_forest(
            rf_model, feat_input, sigma=sigma, temperature=temperature
        )
    ab_softmax = get_prob_classification_forest(
        rf_model, feat_input, sigma=sigma, temperature=temperature
    )

    assert rf_softmax.shape == (1852, 2)
    assert ab_softmax.shape == (1852, 2)


def test_compute_cfe():
    pass
