import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from equalityml import FairML

_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
_MITIGATION_METHODS = ["resampling", "resampling-preferential", "reweighing", "disparate-impact-remover",
                       "correlation-remover"]
_METRICS = ['treatment_equality_ratio', 'treatment_equality_difference', 'balance_positive_class',
            'balance_negative_class', 'equal_opportunity_ratio', 'accuracy_equality_ratio', 'predictive_parity_ratio',
            'predictive_equality_ratio', 'statistical_parity_ratio', 'all']

@pytest.mark.parametrize("estimator", _ESTIMATORS)
def test_bias_mitigation(estimator):
    np.random.seed(0)

    sex_col = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    weight_col = [80, 70, 60, 70, 60, 60, 70, 60, 70, 60]
    target_col = [1, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    training_data = pd.DataFrame({"sex": sex_col, "weight": weight_col, "Y": target_col})

    target_variable = "Y"
    protected_variable = "sex"

    # Fit a machine learning model
    if estimator == SVC:
        _estimator = estimator(probability=True)
    else:
        _estimator = estimator()
    _estimator.fit(training_data.drop(columns=target_variable), target_col)

    fairml = FairML(ml_model=_estimator, training_data=training_data,
                    target_variable=target_variable,
                    protected_variable=protected_variable, privileged_class=1)

    # resampling-uniform
    mitigation_method = "resampling-uniform"
    data_transformed = fairml.bias_mitigation(mitigation_method)

    assert data_transformed.shape == training_data.shape


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("estimator", _ESTIMATORS)
def test_fairness_metric_evaluation(metric, estimator):
    np.random.seed(0)

    sex_col = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]
    weight_col = [80, 70, 60, 70, 60, 60, 70, 60, 70, 60, 60, 70, 70, 70, 60, 60, 60, 60, 70, 60]
    target_col = [1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]
    training_data = pd.DataFrame({"sex": sex_col, "weight": weight_col, "Y": target_col})

    target_variable = "Y"
    protected_variable = "sex"

    # Fit a machine learning model
    if estimator == SVC:
        _estimator = estimator(probability=True)
    else:
        _estimator = estimator()
    _estimator.fit(training_data.drop(columns=target_variable), target_col)

    pred_class = _estimator.predict(training_data.drop(columns=target_variable))
    pred_prob = _estimator.predict_proba(training_data.drop(columns=target_variable))
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # Evaluate some scores
    prev_auc = roc_auc_score(target_col, pred_prob)  # Area under a curve
    prev_accuracy = accuracy_score(target_col, pred_class)  # classification accuracy

    print(prev_auc, prev_accuracy)

    y_pred = pred_prob.copy()
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    tn, fp, fn, tp = confusion_matrix(target_col, y_pred).ravel()
    print(tn, fp, fn, tp)

    fairml = FairML(ml_model=_estimator, training_data=training_data,
                    target_variable=target_variable,
                    protected_variable=protected_variable, privileged_class=1)

    # evaluate fairness
    fairnes_metric = fairml.evaluate_fairness(metric)
    print(fairnes_metric)
