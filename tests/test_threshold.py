import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pytest

from equalityml.fair import FAIR
from equalityml.threshold import discrimination_threshold, binary_threshold_score

_DECISION_THRESHOLD = [['f1', 'max'], ['recall', 'max'], ['precision', 'max'], ['queue_rate', 'limit', '0.1'],
                       ['statistical_parity_ratio', 'max'], ['precision']]

@pytest.fixture()
def dataset():
    np.random.seed(0)

    random_col = np.random.normal(size=30)
    sex_col = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    weight_col = [80, 75, 70, 65, 60, 85, 70, 75, 70, 70, 70, 80, 70, 70, 70, 80, 75, 70, 65, 70,
                  70, 75, 80, 75, 75, 70, 65, 70, 75, 65]
    target_col = [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,
                  0, 1, 0, 1, 1, 0, 0, 1, 1, 0]
    training_data = pd.DataFrame({"random": random_col, "sex": sex_col, "weight": weight_col, "Y": target_col})

    dataset = {"training_data": training_data, "target_variable": "Y", "protected_variable": "sex"}

    return dataset


@pytest.mark.parametrize("decision_maker", _DECISION_THRESHOLD)
def test_discrimination_threshold(dataset, decision_maker):
    np.random.seed(0)

    # Machine learning model
    model = LogisticRegression()

    # FAIR object
    fair_object = FAIR(ml_model=model,
                       training_data=dataset["training_data"],
                       target_variable=dataset["target_variable"],
                       protected_variable=dataset["protected_variable"],
                       privileged_class=1,
                       random_seed=0)

    X = dataset["training_data"].drop(columns=dataset["target_variable"])
    y = dataset["training_data"][dataset["target_variable"]]

    # Compute the optimal discrimination_threshold
    dt = discrimination_threshold(model,
                                  X,
                                  y,
                                  fair_object=fair_object,
                                  metrics=[decision_maker[0]],
                                  decision_maker=decision_maker,
                                  random_seed=0)

    assert 0 <= dt <= 1

    # Just evaluating
    model = LogisticRegression()
    model.fit(X, y)
    dt = discrimination_threshold(model,
                                  X,
                                  y,
                                  fair_object=fair_object,
                                  metrics=[decision_maker[0]],
                                  decision_maker=decision_maker,
                                  model_training=False,
                                  random_seed=0)

    assert 0 <= dt <= 1


@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
def test_binary_threshold_score(dataset, threshold):
    np.random.seed(0)

    X_train = dataset["training_data"].drop(columns=dataset["target_variable"])
    y_train = dataset["training_data"][dataset["target_variable"]]

    # Train a machine learning model
    _estimator = LogisticRegression()
    _estimator.fit(X_train, y_train)

    # F1 score
    f1_score = binary_threshold_score(_estimator, X_train, y_train, scoring="f1", threshold=threshold)

    assert 0 <= f1_score <= 1

    # Accuracy
    accuracy = binary_threshold_score(_estimator, X_train, y_train, scoring="accuracy", threshold=threshold)

    assert 0 <= accuracy <= 1
