import tensorflow as tf
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from src.utils import filter_hinge_loss, calculate_distance


def _parse_class_tree(tree, feat_input, sigma: float) -> list:
    """
    This function traverses the tree structure to compute impurity of each node and
    use sigmoid function to approximate them.
    """
    # Code is adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    nodes = [None] * n_nodes
    leaf_nodes = [[] for _ in range(len(tree.classes_))]

    node_depth = np.zeros(shape=n_nodes, dtype=np.int32)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]

    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        cur_node = nodes[i]
        if children_left[i] != children_right[i]:

            if cur_node is None:
                cur_node = 1.0

            sigma = np.full(len(feat_input), sigma)
            activation = tf.math.sigmoid(
                (feat_input[:, feature[i]] - threshold[i]) * sigma
            )

            left_node, right_node = 1.0 - activation, activation
            nodes[children_left[i]], nodes[children_right[i]] = (
                cur_node * left_node,
                cur_node * right_node,
            )

        else:
            max_class = np.argmax(values[i])
            leaf_nodes[max_class].append(cur_node)

    return leaf_nodes


def get_prob_classification_tree(tree, feat_input, sigma: float) -> tf.Tensor:
    """
    This function takes approximated leaf_nodes' impurity and return each data points' probability
    """

    leaf_nodes = _parse_class_tree(tree, feat_input, sigma)

    if tree.tree_.node_count > 1:
        prob_list = [sum(leaf_nodes[c_i]) for c_i in range(len(tree.classes_))]
        i = 0
        while i < len(prob_list):
            if prob_list[i].numpy().all() == 0:
                prob_list.pop(i)
            else:
                i += 1

        prob_stacked = tf.stack(prob_list, axis=-1)

    else:  # sometimes tree only has one node
        only_class = tree.predict(
            tf.reshape(feat_input[0, :], shape=(1, -1))
        )  # can differ depending on particular samples used to train each tree

        correct_class = tf.constant(
            1, shape=(len(feat_input)), dtype=tf.float32
        )  # prob(belong to correct class) = 100 since there's only one node
        incorrect_class = tf.constant(
            0, shape=(len(feat_input)), dtype=tf.float32
        )  # prob(wrong class) = 0
        if only_class == 1.0:
            class_0 = incorrect_class
            class_1 = correct_class
        elif only_class == 0.0:
            class_0 = correct_class
            class_1 = incorrect_class
        else:
            raise ValueError
        class_labels = [class_0, class_1]
        prob_stacked = tf.stack(class_labels, axis=1)
    return prob_stacked


def get_prob_classification_forest(
    model, feat_input: tf.Tensor, sigma: float, temperature: float
) -> tf.Tensor:
    """
    This function takes decision tree node's probabilities of each data point and calculate softmax
    """
    dt_prob_list = [
        get_prob_classification_tree(estimator, feat_input, sigma)
        for estimator in model.estimators_
    ][:100]

    if isinstance(model, AdaBoostClassifier):
        weights = model.estimator_weights_
    elif isinstance(model, RandomForestClassifier):
        weights = np.full(len(model.estimators_), 1 / len(model.estimators_))

    logits = sum(weight * tree for weight, tree in zip(weights, dt_prob_list))

    temperature = np.full(len(feat_input), temperature)
    if type(temperature) in [float, int]:
        expits = tf.exp(temperature * logits)
    else:
        expits = tf.exp(temperature[:, None] * logits)

    softmax = expits / tf.reduce_sum(expits, axis=1)[:, None]

    return softmax


def compute_cfe(
    model,
    feat_input,
    distance_function,
    opt,
    sigma_val,
    temperature_val,
    distance_weight_val,
    lr,
    num_iter=100,
    x_train=None,
    verbose=1,
):
    perturbed = tf.Variable(
        initial_value=feat_input,
        trainable=True,
        name="perturbed_features",
    )

    n_examples = len(feat_input)
    distance_weight = np.full(n_examples, distance_weight_val)
    to_optimize = [perturbed]
    indicator = np.ones(n_examples)
    best_perturb = np.zeros(perturbed.shape)
    best_distance = np.full(n_examples, 1000.0)  # all distances should be below 1000
    perturb_iteration_found = np.full(n_examples, 1000 * num_iter, dtype=int)

    predictions = model.predict(feat_input)
    class_index = np.zeros(n_examples, dtype=int)
    for i, class_name in enumerate(model.classes_):
        mask = np.equal(predictions, class_name)
        class_index[mask] = i
    class_index = tf.constant(class_index, dtype=tf.int64)
    example_range = tf.constant(np.arange(n_examples, dtype=int))
    example_class_index = tf.stack((example_range, class_index), axis=1)

    if opt == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt == "gd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    with tf.GradientTape(persistent=True) as tape:
        for i in range(num_iter):
            if verbose != 0:
                print(f"iteration {i}")

            hinge_loss = filter_hinge_loss(
                len(model.classes_),
                indicator,
                perturbed,
                sigma_val,
                temperature_val,
                model,
            )

            approx_prob = tf.gather_nd(hinge_loss, example_class_index)

            distance = calculate_distance(
                distance_function, perturbed, feat_input, x_train
            )

            hinge_approx_prob = indicator * approx_prob
            loss = tf.reduce_mean(hinge_approx_prob + distance_weight * distance)

            grad = tape.gradient(loss, to_optimize)

            optimizer.apply_gradients(
                zip(grad, to_optimize),
            )
            perturbed.assign(tf.math.minimum(1, tf.math.maximum(0, perturbed)))

            true_distance = calculate_distance(
                distance_function, perturbed, feat_input, x_train
            )

            cur_predict = model.predict(perturbed.numpy())
            indicator = np.equal(predictions, cur_predict).astype(np.float32)
            idx_flipped = np.argwhere(indicator == 0).flatten()

            # get the best perturbation so far
            mask_flipped = np.not_equal(predictions, cur_predict)

            perturb_iteration_found[idx_flipped] = np.minimum(
                i + 1, perturb_iteration_found[idx_flipped]
            )

            distance_numpy = true_distance.numpy()
            mask_smaller_dist = np.less(
                distance_numpy, best_distance
            )

            temp_dist = best_distance.copy()
            temp_dist[mask_flipped] = distance_numpy[mask_flipped]
            best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

            temp_perturb = best_perturb.copy()
            temp_perturb[mask_flipped] = perturbed[mask_flipped]
            best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

            unchanged_ever = best_distance[best_distance == 1000.0]
            cfe_distance = best_distance[best_distance != 1000.0]

        return unchanged_ever, cfe_distance, best_distance, best_perturb
