import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest

from equalityml import FAIR

_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
_MITIGATION_METHODS = ["resampling", "resampling-preferential", "reweighing", "disparate-impact-remover",
                       "correlation-remover"]
_METRICS = [('treatment_equality_ratio', 3), ('treatment_equality_difference', 0.6666),
            ('balance_positive_class', 0.9307),
            ('balance_negative_class', 0.6830), ('equal_opportunity_ratio', 0.7), ('accuracy_equality_ratio', 1.0),
            ('predictive_parity_ratio', 0.9), ('predictive_equality_ratio', 0.4), ('statistical_parity_ratio', 0.5555)]


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


@pytest.mark.parametrize("mitigation_method", _MITIGATION_METHODS)
@pytest.mark.parametrize("estimator", _ESTIMATORS)
def test_bias_mitigation(dataset, mitigation_method, estimator):
    np.random.seed(0)

    # Train a machine learning model
    if estimator == SVC:
        _estimator = estimator(probability=True)
    else:
        _estimator = estimator()

    _estimator.fit(dataset["training_data"].drop(columns=dataset["target_variable"]),
                   dataset["training_data"][dataset["target_variable"]])

    fair_object = FAIR(ml_model=_estimator, training_data=dataset["training_data"],
                       target_variable=dataset["target_variable"],
                       protected_variable=dataset["protected_variable"], privileged_class=1)

    # call bias mitigation method
    mitigation_result = fair_object.bias_mitigation(mitigation_method)
    if mitigation_method == "reweighing":
        mitigated_weights = mitigation_result['weights']
        assert len(mitigated_weights) == dataset["training_data"].shape[0]
        assert all(isinstance(weight, float) for weight in mitigated_weights)
    else:
        mitigated_data = mitigation_result['training_data']
        assert mitigated_data.shape == dataset["training_data"].shape


@pytest.mark.parametrize("metric, estimated_value", _METRICS)
def test_fairness_metric_evaluation(dataset, metric, estimated_value):
    np.random.seed(0)

    # Train a LogisticRegression model
    _estimator = LogisticRegression()
    _estimator.fit(dataset["training_data"].drop(columns=dataset["target_variable"]),
                   dataset["training_data"][dataset["target_variable"]])

    fair_object = FAIR(ml_model=_estimator, training_data=dataset["training_data"],
                       target_variable=dataset["target_variable"],
                       protected_variable=dataset["protected_variable"], privileged_class=1)

    # Compute fairness metric
    fairness_metric = fair_object.fairness_metric(metric)
    assert np.allclose(fairness_metric[metric], estimated_value, rtol=1.e-3)


@pytest.mark.parametrize("mitigation_method",
                         ["resampling", "resampling-preferential", "reweighing", "disparate-impact-remover"])
def test_workflow(dataset, mitigation_method):
    np.random.seed(0)

    X_train = dataset["training_data"].drop(columns=dataset["target_variable"])
    y_train = dataset["training_data"][dataset["target_variable"]]

    # Train a machine learning model
    _estimator = LogisticRegression()
    _estimator.fit(X_train, y_train)

    fair_object = FAIR(ml_model=_estimator, training_data=dataset["training_data"],
                       target_variable=dataset["target_variable"],
                       protected_variable=dataset["protected_variable"], privileged_class=1)

    # Compute fairness metric
    metric = "statistical_parity_ratio"
    prev_fairness_metric = fair_object.fairness_metric(metric)

    # Apply a bias mitigation method
    mitigation_result = fair_object.bias_mitigation(mitigation_method)
    if mitigation_method == "reweighing":
        mitigated_weights = mitigation_result['weights']
        # Re-train the machine learning model using mitigated weights
        _estimator.fit(X_train, y_train, sample_weight=mitigated_weights)
    else:
        mitigated_data = mitigation_result['training_data']
        X_train = mitigated_data.drop(columns=dataset["target_variable"])
        y_train = mitigated_data[dataset["target_variable"]]

        # Re-train the machine learning model based on mitigated data
        _estimator.fit(X_train, y_train)

    fair_object.update_classifier(_estimator)
    fairness_metric = fair_object.fairness_metric(metric)

    # The new fairness metric value shall be better than previous one
    assert prev_fairness_metric[metric] < fairness_metric[metric] < 1
