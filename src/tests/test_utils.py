import pytest
import numpy as np
import pandas as pd
import pickle
from src.utils import (
    filter_hinge_loss,
    safe_euclidean,
    true_euclidean,
    safe_cosine,
    true_cosine,
    safe_l1,
    true_l1,
    safe_mahal,
    true_mahal,
    tf_cov,
    safe_open,
    mkdir_p,
    calculate_distance,
)

dt_model = pickle.load(open("my_models/dt_cf_compas_num_data_train.pkl", "rb"))
rf_model = pickle.load(open("my_models/rf_cf_compas_num_data_train.pkl", "rb"))
ab_model = pickle.load(open("my_models/ab_cf_compas_num_data_train.pkl", "rb"))
df = pd.read_csv("data/cf_compas_num_data_test.tsv", sep="\t", index_col=0)
feat_input = df.values.astype(float)[:, :-1]
sigma = 5.0
temperature = 10.0
indicator = np.zeros(len(df))


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


def test_safe_euclidean():
    assert safe_euclidean(feat_input).shape == (1852,)


def test_true_euclidean():
    assert true_euclidean(feat_input).shape == (1852,)


def test_safe_cosine():
    assert safe_cosine(feat_input, feat_input).shape == (1852,)


def test_true_cosine():
    assert true_cosine(feat_input, feat_input).shape == (1852,)


def test_safe_l1():
    assert safe_l1(feat_input).shape == (1852,)


def test_true_l1():
    assert true_l1(feat_input).shape == (1852,)


def test_tf_cov():
    assert tf_cov(feat_input).shape == (6, 6)


def test_safe_mahal():
    assert safe_mahal(feat_input, feat_input).shape == (1852,)


def test_true_mahal():
    assert true_mahal(feat_input, feat_input).shape == (1852,)


def test_mkdir_p():
    pass


def test_safe_open():
    pass


def test_calculate_distance():
    assert calculate_distance(
        distance_function="euclidean", perturbed=feat_input, feat_input=feat_input
    ).shape == (1852,)
    assert calculate_distance(
        distance_function="cosine", perturbed=feat_input, feat_input=feat_input
    ).shape == (1852,)
    assert calculate_distance(
        distance_function="l1", perturbed=feat_input, feat_input=feat_input
    ).shape == (1852,)
    assert calculate_distance(
        distance_function="mahal",
        perturbed=feat_input,
        feat_input=feat_input,
        x_train=feat_input
    ).shape == (1852,)
