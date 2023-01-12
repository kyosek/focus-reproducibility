import tensorflow as tf
import os
import errno


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
    dist = 1 - cosine_loss(normalize_x1, normalize_x2) + epsilon

    dist = tf.cast(tf.squeeze(dist), tf.float32)
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
    dist = tf.cast(dist, tf.float32)
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


def safe_mahal(x_test, x_train, epsilon=10.0 ** -10):
    covar = tf_cov(x_train)
    inv_covar = tf.linalg.inv(covar)
    return tf.reduce_sum(
        tf.multiply(tf.matmul(x_test + epsilon, inv_covar), x_test + epsilon), axis=1
    )


def true_mahal(x_test, x_train):
    covar = tf_cov(x_train)
    inv_covar = tf.linalg.inv(covar)
    return tf.reduce_sum(tf.multiply(tf.matmul(x_test, inv_covar), x_test), axis=1)


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


def calculate_distance(distance_function: str, perturbed, feat_input, x_train=None):
    if distance_function == "euclidean":
        return safe_euclidean(perturbed - feat_input, axis=1)
    elif distance_function == "cosine":
        return safe_cosine(perturbed, feat_input)
    elif distance_function == "l1":
        return safe_l1(perturbed - feat_input)
    elif distance_function == "mahal":
        try:
            x_train.any()
            return safe_mahal(perturbed - feat_input, x_train)
        except ValueError:
            print("x_train is empty")
