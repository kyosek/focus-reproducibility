import numpy as np
import pandas as pd
import tensorflow as tf

import pytest

from sklearn.metrics.pairwise import cosine_similarity

from src.utils import (
    safe_euclidean,
    safe_cosine,
    safe_l1,
    safe_mahal,
    tf_cov,
    calculate_distance,
)

compas_path = "data/cf_compas_num_data_test.tsv"
heloc_path = "data/cf_heloc_data_test.tsv"
shop_path = "data/cf_shop2_data_test.tsv"
wine_path = "data/cf_wine_data_test.tsv"

epsilon = 10.0 ** -10

covariance_test_data = [
    (
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        tf.constant(
            [[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]], dtype=tf.float64
        ),
    ),
    (
        tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float64),
        tf.convert_to_tensor(
            np.cov(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T, bias=True),
            dtype=tf.float64,
        ),
    ),
    # COMPAS dataset
    (
        pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        tf.convert_to_tensor(
            np.cov(
                pd.read_csv(compas_path, sep="\t", index_col=0)
                .values.astype(float)[:, :-1]
                .T,
                bias=True,
            ),
            dtype=tf.float64,
        ),
    ),
    # HELOC dataset
    (
        pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        tf.convert_to_tensor(
            np.cov(
                pd.read_csv(heloc_path, sep="\t", index_col=0)
                .values.astype(float)[:, :-1]
                .T,
                bias=True,
            ),
            dtype=tf.float64,
        ),
    ),
    # Shopping dataset
    (
        pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        tf.convert_to_tensor(
            np.cov(
                pd.read_csv(shop_path, sep="\t", index_col=0)
                .values.astype(float)[:, :-1]
                .T,
                bias=True,
            ),
            dtype=tf.float64,
        ),
    ),
    # Wine dataset
    (
        pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        tf.convert_to_tensor(
            np.cov(
                pd.read_csv(wine_path, sep="\t", index_col=0)
                .values.astype(float)[:, :-1]
                .T,
                bias=True,
            ),
            dtype=tf.float64,
        ),
    ),
]

distance_test_data = [
    # COMPAS dataset
    (
        pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        pd.read_csv(compas_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    ),
    # HELOC dataset
    (
        pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        pd.read_csv(heloc_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    ),
    # Shopping dataset
    (
        pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        pd.read_csv(shop_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    ),
    # Wine dataset
    (
        pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
        pd.read_csv(wine_path, sep="\t", index_col=0).values.astype(float)[:, :-1],
    ),
]


@pytest.mark.parametrize("feat_input_cov, expected_output_cov", covariance_test_data)
def test_tf_cov(feat_input_cov, expected_output_cov):
    assert tf_cov(feat_input_cov).numpy().all() == expected_output_cov.numpy().all()


@pytest.mark.parametrize("feat_input, expected_output", distance_test_data)
def test_safe_euclidean(feat_input, expected_output):
    expected = (np.sum(feat_input ** 2, axis=-1) + epsilon) ** 0.5
    assert safe_euclidean(feat_input).numpy().all() == expected.all()


@pytest.mark.parametrize("feat_input, expected_output", distance_test_data)
def test_safe_cosine(feat_input, expected_output):
    expected = 1 - cosine_similarity(feat_input, feat_input) + epsilon
    assert safe_cosine(feat_input, feat_input, epsilon).numpy().all() == expected.all()


@pytest.mark.parametrize("feat_input, expected_output", distance_test_data)
def test_safe_l1(feat_input, expected_output):
    expected = np.sum(abs(feat_input), axis=1) + epsilon
    assert safe_l1(feat_input).numpy().all() == expected.all()


@pytest.mark.parametrize("feat_input, expected_output", distance_test_data)
def test_safe_mahal(feat_input, expected_output):

    x_mu = feat_input - np.mean(feat_input)
    cov = np.cov(feat_input.T)
    inv_covar = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covar)
    mahal = np.dot(left, x_mu.T)
    expected = mahal.diagonal()

    assert safe_mahal(feat_input, feat_input).numpy().all() == expected.all()


@pytest.mark.parametrize("feat_input, expected_output", distance_test_data)
def test_calculate_distance(feat_input, expected_output):
    assert (
        calculate_distance(
            distance_function="euclidean", perturbed=feat_input, feat_input=feat_input
        ).shape
        == expected_output
    )
    assert (
        calculate_distance(
            distance_function="cosine", perturbed=feat_input, feat_input=feat_input
        ).shape
        == expected_output
    )
    assert (
        calculate_distance(
            distance_function="l1", perturbed=feat_input, feat_input=feat_input
        ).shape
        == expected_output
    )
    assert (
        calculate_distance(
            distance_function="mahal",
            perturbed=feat_input,
            feat_input=feat_input,
            x_train=feat_input,
        ).shape
        == expected_output
    )


def test_mkdir_p():
    # No need to test this function
    pass


def test_safe_open():
    # No need to test this function
    pass
