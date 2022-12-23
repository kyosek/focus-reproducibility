import tensorflow as tf
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


# def _exact_activation_by_index(feat_input, feat_index, threshold):
#     boolean_act = tf.math.greater(feat_input[:, feat_index], threshold)
#     return tf.logical_not(boolean_act), boolean_act


# def _approx_activation_by_index(feat_input, feat_index, threshold, sigma):
#     """
#     Looking into each feature values against the threshold
#     and put it through the sigmoid function
#     returns left and right children approximation
#
#     left child = sig(threshold - feat_input)
#     right child = sig(feat_input - threshold)
#     sig(z) = 1 / (1 + exp(sigma * z))
#     when sigma increases, the approximation is closer to t_j
#     """
#     activation = tf.math.sigmoid((feat_input[:, feat_index] - threshold) * sigma)
#
#     return 1.0 - activation, activation


# def _double_activation_by_index(feat_input, feat_index, threshold, sigma):
#     e_l, e_r = _exact_activation_by_index(feat_input, feat_index, threshold)
#     a_l, a_r = _approx_activation_by_index(feat_input, feat_index, threshold, sigma)
#     return (e_l, a_l), (e_r, a_r)
#
#
# def _split_node_by_index(node, feat_input, feat_index, threshold, sigma):
#     # exact node and approximate node
#     e_o, a_o = node
#     ((e_l, a_l), (e_r, a_r)) = _double_activation_by_index(
#         feat_input, feat_index, threshold, sigma
#     )
#     return (
#         (tf.logical_and(e_l, e_o), a_l * a_o),
#         (tf.logical_and(e_r, e_o), a_r * a_o),
#     )


# def _split_exact(node, feat_input, feat_index, threshold):
#     print("split exact")
#     if node is None:
#         node = True
#     l_n, r_n = _exact_activation_by_index(feat_input, feat_index, threshold)
#     return tf.logical_and(node, l_n), tf.logical_and(node, r_n)


def _parse_class_tree(tree, feat_input, sigma: float):
    # Code is adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    # n_classes = len(tree.classes_)
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    # feature returns the nodes/leaves in a sequential order as a DFS
    feature = tree.tree_.feature
    # threshold for impurity - do they use gain instead?
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    nodes = [None] * n_nodes
    leaf_nodes = [[] for _ in range(len(tree.classes_))]

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes):
        cur_node = nodes[i]
        #  If a split node, get nodes updated
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


def get_prob_classification_tree(tree, feat_input, sigma: float):

    leaf_nodes = _parse_class_tree(tree, feat_input, sigma)

    if tree.tree_.node_count > 1:
        prob_list = [sum(leaf_nodes[c_i]) for c_i in range(len(tree.classes_))]
        i = 0
        while i < len(prob_list):
            if prob_list[i].numpy().all() == 0:
                prob_list.pop(i)
            else:
                i += 1

        stacked = tf.stack(prob_list, axis=-1)

    else:  # sometimes tree only has one node
        only_class = tree.predict(
            tf.reshape(feat_input[0, :], shape=(1, -1))
        )  # can differ depending on particular samples used to train each tree

        correct_class = tf.constant(
            1, shape=(len(feat_input)), dtype=tf.float64
        )  # prob(belong to correct class) = 100 since there's only one node
        incorrect_class = tf.constant(
            0, shape=(len(feat_input)), dtype=tf.float64
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
        stacked = tf.stack(class_labels, axis=1)
    return stacked


# def get_exact_classification_tree(tree, feat_input, sigma):
#     leaf_nodes = _parse_class_tree(tree, feat_input, sigma)
#
#     out_l = []
#     for class_name in tree.classes_:
#         out_l.append(tf.reduce_any(leaf_nodes[class_name]))
#     return tf.cast(tf.stack(out_l, axis=-1), dtype=tf.float64)


def get_prob_classification_forest(model, feat_input: tf.Tensor, sigma: float, temperature: float):
    dt_prob_list = [
        get_prob_classification_tree(estimator, feat_input, sigma)
        for estimator in model.estimators_
    ][:100]

    if isinstance(model, AdaBoostClassifier):
        weights = model.estimator_weights_
    elif isinstance(model, RandomForestClassifier):
        weights = np.full(len(model.estimators_), 1 / len(model.estimators_))

    logits = sum(w * tree for w, tree in zip(weights, dt_prob_list))

    temperature = np.full(len(feat_input), temperature)
    if type(temperature) in [float, int]:
        expits = tf.exp(temperature * logits)
    else:
        expits = tf.exp(temperature[:, None] * logits)

    softmax = expits / tf.reduce_sum(expits, axis=1)[:, None]

    return softmax
