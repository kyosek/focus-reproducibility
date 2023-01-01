import pytest
import tensorflow as tf
import numpy as np

from src.approximation import get_prob_classification_tree, get_prob_classification_forest
from src.utils import filter_hinge_loss, safe_euclidean, true_euclidean, safe_cosine, true_cosine, safe_l1, true_l1, safe_mahal, true_mahal
from sklearn.tree import DecisionTreeClassifier


def test_filter_hinge_loss():
    n_input = 1000

    pass


def test_safe_euclidean():
    pass


def test_true_euclidean():
    pass
