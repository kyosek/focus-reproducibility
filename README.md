# [Re]:FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

This repository presents a re-implemented codebase and experiment results of [AAAI 2022 paper "FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles"](https://arxiv.org/abs/1911.12199).

## Requirements
Python 3.7

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage
### Generate counterfactual explanation
The main function - run counterfactual explanation generation and save results - is located in src/main.py.
This can be run with arguments:
- model_type: (str) name of the model type (one of "dt", "rf" and "ab")
- num_iter: (int) number of iterations (all the experiments used 1,000)
- sigma: (float) smoothing function of the sigmoid function
- temperature: (float) temperature value for hinge loss
- distance_weight: (float) weight value for distance loss
- lr: (float) learning rate of the optimiser
- opt: (str) name of the optimiser. Use either "adam" or "sd". All the experiments used adam
- data_name: (str) name of the data. All the data are located in /data/
- distance_function: (str) distance function - one of "euclidean", "cosine", "l1" and "mahal"

Below is an example command on the root directory:
```text
python src/main.py model_type=dt num_itr=1000 sigma=10.0 temperature=1.0 weight_distance=0.01 lr=0.001 opt=adam data_name=cf_german_test distance_function=l1
```
>📋  This will create another folder in the main directory called 'results', where the results files will be
> stored in "results" folder.

### Hyperparameter tuning
The hyperparameter tuning module - this will run Optuna's Bayesian optimisation on the predefined search space.
This can be run with the same arguments as the main function + number of trials.
- n_trials: (int) number of hyperparameter tuning trials

Below is an example command on the root directory:
```text
python src/hyperparameter_tuning.py model_type=dt num_itr=1000 sigma=10.0 temperature=1.0 weight_distance=0.01 lr=0.001 opt=adam data_name=cf_german_test distance_function=l1 n_trials=100
```

### Testing
The unit tests can be found in /tests/. This uses pytest to run. You can launch the test suite on the root directory:
```text
pytest
```

## Other python files
### train
train.py file was used to retrain the original models with new hyperparameters.
This file also evaluates accuracy of the retrained model.

Below is an example command on the root directory:
```text
python src/train.py model_type=ab data_name=cf_heloc_data_train max_depth=3 n_estimators=100
```

### Preprocess German dataset
preprocess_german_data.py preprocesses the German credit dataset. This file can be run without any arguments.
```text
python src/preprocess_german_data.py
```

## Data and Models

The datasets and models that were used in the experiments are also available in this repository:
- **Dataset**: data/
- **Pretrained models of the original paper** (just as a reference - not compatible with this code implementation): models/
- **Pretrained models of the reproducibility challenge**: retrained_models/
