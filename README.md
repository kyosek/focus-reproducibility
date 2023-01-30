# [Re]:FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles

This repository is the reproduction study for the [AAAI 2022 paper "FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles"](https://arxiv.org/abs/1911.12199). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

### Generate counterfactual explanation
The main function - generating counterfactual explanations is in src/main.py.
This can be run with arguments:
- sigma: 
Below is an example command on the root directory:
```text
python src/main.py model_type=dt num_itr=1000 sigma=10.0 temperature=1.0 weight_distance=0.01 lr=0.001 opt=adam data_name=cf_german_test distance_function=l1
```
>📋  This will create another folder in the main directory called 'results', where the results files will be stored.

### Hyperparameter tuning
Run below example command on the root directory:
```text
python src/hyperparameter_tuning.py model_type=dt num_itr=1000 sigma=10.0 temperature=1.0 weight_distance=0.01 lr=0.001 opt=adam data_name=cf_german_test distance_function=l1 n_trials=100
```

## Testing
After installation, you can launch the test suite from outside the source directory (you will need to have pytest installed):
```text
pytest
```

## Data and Models

The datasets and models that were used in the experiments are also available in this repository:
- Dataset: /data/
- Pretrained models of the original paper: /models/
- Pretrained models of the reproducibility challenge: /retrained_models/
