import tensorflow as tf
import numpy as np

import trees
from sklearn.tree import DecisionTreeClassifier
import os
import errno


def filter_hinge_loss(
    n_class, mask_vector, feat_input, sigma, temperature, model
) -> tf.Tensor:
    """
    Return hinge loss from input features?
    """
    n_input = feat_input.shape[0]

    filtered_input = tf.boolean_mask(feat_input, mask_vector)

    if not isinstance(model, DecisionTreeClassifier):
        filtered_loss = trees.get_prob_classification_forest(
            model, filtered_input, sigma=sigma, temperature=temperature
        )
    elif isinstance(model, DecisionTreeClassifier):
        filtered_loss = trees.get_prob_classification_tree(model, filtered_input, sigma)

    indices = np.where(mask_vector)[0]
    hinge_loss = tf.tensor_scatter_nd_add(
        np.zeros((n_input, n_class)),
        indices[:, None],
        filtered_loss,
    )
    return hinge_loss


def safe_euclidean(x, epsilon=10.0 ** -10, axis=-1):
    return (tf.reduce_sum(x ** 2, axis=axis) + epsilon) ** 0.5


def true_euclidean(x, axis=-1):
    return (tf.reduce_sum(x ** 2, axis=axis)) ** 0.5


def safe_cosine(x1, x2, epsilon=10.0 ** -10):
    normalize_x1 = tf.nn.l2_normalize(x1, dim=1)
    normalize_x2 = tf.nn.l2_normalize(x2, dim=1)
    cosine_loss = tf.keras.losses.CosineSimilarity(
        axis=-1,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    dist = cosine_loss(normalize_x1, normalize_x2) + 1 + epsilon

    dist = tf.squeeze(dist)
    dist = tf.cast(dist, tf.float64)
    return dist


def true_cosine(x1: object, x2: object) -> object:
    normalize_x1 = tf.nn.l2_normalize(x1, dim=1)
    normalize_x2 = tf.nn.l2_normalize(x2, dim=1)
    cosine_loss = tf.keras.losses.CosineSimilarity(
        axis=-1,
        reduction=tf.keras.losses.Reduction.NONE,
    )
    dist = cosine_loss(normalize_x1, normalize_x2) + 1

    dist = tf.squeeze(dist)
    dist = tf.cast(dist, tf.float64)
    return dist


def safe_l1(x, epsilon=10.0 ** -10, axis=1):
    return tf.reduce_sum(tf.abs(x), axis=axis) + epsilon


def true_l1(x, axis=1):
    return tf.reduce_sum(tf.abs(x), axis=axis)


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float64)
    cov_xx = vx - mx
    return cov_xx


def safe_mahal(x, inv_covar, epsilon=10.0 ** -10):
    return tf.reduce_sum(
        tf.multiply(tf.matmul(x + epsilon, inv_covar), x + epsilon), axis=1
    )


def true_mahal(x, inv_covar):
    return tf.reduce_sum(tf.multiply(tf.matmul(x, inv_covar), x), axis=1)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def safe_open(path, w):
    """ Open "path" for writing, creating any parent directories as needed."""
    mkdir_p(os.path.dirname(path))
    return open(path, w)


def calculate_distance(distance_function: str, perturbed, feat_input, inv_covar=None):
    if distance_function == "euclidean":
        return safe_euclidean(perturbed - feat_input, axis=1)
    elif distance_function == "cosine":
        return safe_cosine(perturbed, feat_input)
    elif distance_function == "l1":
        return safe_l1(perturbed - feat_input)
    elif distance_function == "mahal":
        # if inv_covar not None:
        return safe_mahal(perturbed - feat_input, inv_covar)
        # else:
        #     raise ValueError

