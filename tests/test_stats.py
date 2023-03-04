import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest

from equalityml.fair import FAIR
from equalityml.stats import paired_ttest, mcnemar_table

_ESTIMATORS = [SVC, DecisionTreeClassifier, RandomForestClassifier]
_MITIGATION_METHODS = ["resampling", "resampling-preferential", "reweighing", "disparate-impact-remover",
                       "correlation-remover"]
_THRESHOLDS = [(0.2, [[20, 0], [4, 6]]),
               (0.4, [[22, 0], [7, 1]]),
               (0.6, [[18, 0], [12, 0]]),
               (0.8, [[15, 0], [12, 3]])
               ]


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
    X = training_data.drop(columns="Y")
    y = training_data["Y"]

    dataset = {"training_data": training_data, "target_variable": "Y", "X": X, "y": y, "protected_variable": "sex"}

    return dataset


@pytest.mark.parametrize("threshold, tb", _THRESHOLDS)
def test_mcnemar_table(dataset, threshold, tb):
    np.random.seed(0)

    # Train a 'reference' machine learning model
    model_1 = LogisticRegression()
    model_1.fit(dataset["X"], dataset["y"])

    # Train a second machine learning model
    model_2 = RandomForestClassifier()
    model_2.fit(dataset["X"], dataset["y"])

    table = mcnemar_table(model_1,
                          model_2,
                          dataset["X"],
                          dataset["y"],
                          threshold=threshold)

    assert (table == tb).all()


@pytest.mark.parametrize("estimator", _ESTIMATORS)
@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
def test_mcnemar(dataset, estimator, threshold):
    np.random.seed(0)

    # Train a 'reference' machine learning model
    model_1 = LogisticRegression()
    model_1.fit(dataset["X"], dataset["y"])

    # Train a second machine learning model
    if estimator == SVC:
        model_2 = estimator(probability=True)
    else:
        model_2 = estimator()

    model_2.fit(dataset["X"], dataset["y"])

    results = paired_ttest(model_1,
                           dataset["X"],
                           dataset["y"],
                           model_2=model_2,
                           method="mcnemar",
                           threshold=threshold)

    assert len(results) == 2
    assert 0 <= results[1] <= 1.


@pytest.mark.parametrize("estimator", _ESTIMATORS)
@pytest.mark.parametrize("threshold", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("mitigation_method", _MITIGATION_METHODS)
def test_5x2cv(dataset, estimator, threshold, mitigation_method):
    np.random.seed(0)

    # Reference machine learning model
    model_1 = LogisticRegression()
    model_1.fit(dataset["X"], dataset["y"])

    # Second machine learning model
    if estimator == SVC:
        model_2 = estimator(probability=True)
    else:
        model_2 = estimator()

    fair_object = FAIR(ml_model=model_1,
                       training_data=dataset["training_data"],
                       target_variable=dataset["target_variable"],
                       protected_variable=dataset["protected_variable"],
                       privileged_class=1,
                       random_seed=0)

    results = paired_ttest(model_1,
                           dataset["X"],
                           dataset["y"],
                           method="5x2cv",
                           fair_object=fair_object,
                           mitigation_method=mitigation_method,
                           threshold=threshold,
                           random_seed=0)

    assert len(results) == 2
    assert 0 <= results[1] <= 1.

    results = paired_ttest(model_1,
                           dataset["X"],
                           dataset["y"],
                           model_2=model_2,
                           method="5x2cv",
                           threshold=threshold,
                           random_seed=0)

    assert len(results) == 2
    assert 0 <= results[1] <= 1.
