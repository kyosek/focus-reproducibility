import numpy as np
import pandas as pd
import pickle

from src.approximation import (
    _parse_class_tree,
    get_prob_classification_tree,
    get_prob_classification_forest,
    filter_hinge_loss,
    compute_cfe,
)

dt_model = pickle.load(
    open("my_models/dt_cf_compas_num_data_train_replication.pkl", "rb")
)
rf_model = pickle.load(
    open("my_models/rf_cf_compas_num_data_train_replication.pkl", "rb")
)
ab_model = pickle.load(
    open("my_models/ab_cf_compas_num_data_train_replication.pkl", "rb")
)
df = pd.read_csv("data/cf_compas_num_data_test.tsv", sep="\t", index_col=0)
feat_input = df.values.astype(float)[:, :-1]
sigma = 5.0
temperature = 10.0
indicator = np.zeros(len(df))


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


def test_filter_hinge_loss():
    dt_hinge_loss = filter_hinge_loss(
        len(dt_model.classes_),
        indicator,
        feat_input,
        sigma,
        temperature,
        dt_model,
    )
    rf_hinge_loss = filter_hinge_loss(
        len(dt_model.classes_),
        indicator,
        feat_input,
        sigma,
        temperature,
        rf_model,
    )
    ab_hinge_loss = filter_hinge_loss(
        len(dt_model.classes_),
        indicator,
        feat_input,
        sigma,
        temperature,
        ab_model,
    )

    assert dt_hinge_loss.shape == (1852, 2)
    assert rf_hinge_loss.shape == (1852, 2)
    assert ab_hinge_loss.shape == (1852, 2)


def test_compute_cfe():
    pass
