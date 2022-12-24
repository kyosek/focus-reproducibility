import utils
import tensorflow as tf
import numpy as np
from utils import calculate_distance, filter_hinge_loss
from evaluate import generate_perturbed_df, generate_perturbed_df_diff, generate_cf_stats
import joblib
from sklearn.tree import DecisionTreeClassifier
import argparse
import time
import pandas as pd
import json


# parser = argparse.ArgumentParser()
# parser.add_argument("--sigma", type=float, required=False, default=1.0)
# parser.add_argument("--temperature", type=float, required=False, default=1.0)
# parser.add_argument("--distance_weight", type=float, required=False, default=0.01)
# parser.add_argument("--lr", type=float, required=False, default=0.001)
# parser.add_argument(
#     "--opt",
#     type=str,
#     required=False,
#     default="adam",
#     help="Options are either adam or gd (as str)",
# )
# parser.add_argument("--model_name", type=str, required=True)
# parser.add_argument("--data_name", type=str, required=True)
# parser.add_argument("--model_type", type=str, required=True)
# parser.add_argument("--distance_function", type=str, required=True)

# args = parser.parse_args()
# sigma_val = args.sigma
# temperature_val = args.temperature
# distance_weight_val = args.distance_weight
# lr = args.lr
# opt = args.opt
# model_name = args.model_name
# data_name = args.data_name
# model_type = args.model_type
# distance_function = args.distance_function

sigma_val = 4.0
temperature_val = 2.0
distance_weight_val = 0.005
lr = 0.001
opt = "adam"
num_iter = 10
distance_function = "mahal"

model_name = "model_dt_cf_compas_num_depth4"
data_name = "cf_compas_num_data_test.tsv"
model_type = "ss"

start_time = time.time()
# had to match the scikit-learn version to 0.21.3 in order to load the model but eventually upgrade it
model = joblib.load("models/{}".format(model_name), "rb")

df = pd.read_csv("data/{}".format(data_name), sep="\t", index_col=0)
feat_columns = df.columns
feat_matrix = df.values.astype(float)

n_examples = feat_matrix.shape[0]
n_class = len(model.classes_)

# Remove the last column which is the label
feat_input = feat_matrix[:, :-1]

predictions = model.predict(feat_input)
class_index = np.zeros(n_examples, dtype=int)
for i, class_name in enumerate(model.classes_):
    mask = np.equal(predictions, class_name)
    class_index[mask] = i
class_index = tf.constant(class_index, dtype=tf.int64)
example_range = tf.constant(np.arange(n_examples, dtype=int))
example_class_index = tf.stack((example_range, class_index), axis=1)

# Include training data to compute covariance matrix for Mahalanobis distance
# sort this when refactoring the inputs
train_name = data_name.replace("test", "train")
train_data = pd.read_csv("data/{}".format(train_name), sep="\t", index_col=0)
x_train = np.array(train_data.iloc[:, :-1])
covar = utils.tf_cov(x_train)
inv_covar = tf.linalg.inv(covar)

perturbed = tf.Variable(
    initial_value=feat_input,
    trainable=True,
    name="perturbed_features",
)

output_root = (
    "hyperparameter_tuning/{}/{}/{}/perturbs_{}_sigma{}_temp{}_dweight{}_lr{}".format(
        distance_function,
        data_name,
        model_type,
        opt,
        sigma_val,
        temperature_val,
        distance_weight_val,
        lr,
    )
)

distance_weight = np.full(n_examples, distance_weight_val)
to_optimize = [perturbed]
indicator = np.ones(n_examples)
best_perturb = np.zeros(perturbed.shape)
best_distance = np.full(n_examples, 1000.0)  # all distances should be below 1000
perturb_iteration_found = np.full(n_examples, 1000 * num_iter, dtype=int)
average_distance = np.zeros(num_iter)

# calling optimizer in the for loop will change the results
if opt == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
elif opt == "gd":
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

with tf.GradientTape(persistent=True) as tape:
    for i in range(num_iter):
        print(f"iteration {i}")

        hinge_loss = filter_hinge_loss(
            n_class, indicator, perturbed, sigma_val, temperature_val, model
        )
        approx_prob = tf.gather_nd(hinge_loss, example_class_index)

        distance = calculate_distance(
            distance_function, perturbed, feat_input, inv_covar
        )

        hinge_approx_prob = indicator * approx_prob
        loss = tf.reduce_mean(hinge_approx_prob + distance_weight * distance)

        grad = tape.gradient(loss, to_optimize)

        optimizer.apply_gradients(
            zip(grad, to_optimize),
        )
        # Make sure perturbed values are between 0 and 1 (inclusive)
        perturbed.assign(tf.math.minimum(1, tf.math.maximum(0, perturbed)))

        true_distance = calculate_distance(
            distance_function, perturbed, feat_input, inv_covar
        )

        cur_predict = model.predict(perturbed.numpy())
        indicator = np.equal(predictions, cur_predict).astype(np.float64)
        idx_flipped = np.argwhere(indicator == 0).flatten()

        # get the best perturbation so far
        mask_flipped = np.not_equal(predictions, cur_predict)

        perturb_iteration_found[idx_flipped] = np.minimum(
            i + 1, perturb_iteration_found[idx_flipped]
        )

        distance_numpy = true_distance.numpy()
        mask_smaller_dist = np.less(
            distance_numpy, best_distance
        )  # is dist < previous best dist?

        temp_dist = best_distance.copy()
        temp_dist[mask_flipped] = distance_numpy[mask_flipped]
        best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

        temp_perturb = best_perturb.copy()
        temp_perturb[mask_flipped] = perturbed[mask_flipped]
        best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

        end_time = time.time()

        unchanged_ever = best_distance[best_distance == 1000.0]
        counterfactual_examples = best_distance[best_distance != 1000.0]
        average_distance[i] = np.mean(counterfactual_examples)

perturb_iteration_found[perturb_iteration_found == 1000 * num_iter] = 0

# Evaluation
generate_cf_stats(output_root, data_name, distance_function, unchanged_ever, counterfactual_examples)
df_perturb = generate_perturbed_df(n_examples, best_distance, best_perturb, feat_columns)
df_diff = generate_perturbed_df_diff(df_perturb, feat_input)

print("Finished!! ~{} sec".format(np.round(end_time - start_time), 2))
