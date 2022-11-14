import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest

from fairml import BiasMitigation
from fairml import FairnessMetric

_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
_MITIGATORS = ["resampling", "resampling-preferential", "reweighing", "disparate-impact-remover",
               "correlation-remover"]


@pytest.mark.parametrize("mitigator", _MITIGATORS)
@pytest.mark.parametrize("estimator", _ESTIMATORS)
def test_smoke(mitigator, estimator):
    np.random.seed(0)
    n = 100
    X0 = np.random.normal(size=n)
    X1 = np.random.choice([1, 2], size=n)
    Y = np.random.choice([0, 1], size=n)
    df = pd.DataFrame({"X0": X0, "X1": X1, "Y": Y})

    _estimator = estimator()
    _estimator.fit(df.drop(columns="Y"), Y)

    bias_mitigation = BiasMitigation(ml_model=_estimator, data=df,
                                     target_attribute="Y",
                                     protected_attribute="X1", privileged_class=1)

    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigator)


TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(TESTS_PATH, os.pardir))

mitigation_methods = ("resampling", "resampling-preferential", "reweighing", "disparate-impact-remover",
                      "correlation-remover")


@pytest.fixture()
def train_ml_model():
    # Read training and testing data.
    target_var = "HOS"
    train_path = os.path.join(PACKAGE_PATH, 'data', 'data_train.csv')
    training_data = pd.read_csv(train_path)
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    test_path = os.path.join(PACKAGE_PATH, 'data', 'data_test.csv')
    testing_data = pd.read_csv(test_path)

    # Train a machine learning model
    ml_model = LogisticRegression()
    ml_model.fit(X_train, y_train)

    return {"ml_model": ml_model, "training_data": training_data, "testing_data": testing_data,
            "target_var": target_var}


# =======================================================


@pytest.mark.parametrize("mitigation_method", mitigation_methods)
def test_bias_mitigation(train_ml_model, mitigation_method):
    protected_attribute = 'RACERETH'

    # Compute Fairness score for "statistical_parity_ratio"
    fairness_metric = FairnessMetric(ml_model=train_ml_model["ml_model"], data=train_ml_model["testing_data"],
                                     target_attribute=train_ml_model["target_var"],
                                     protected_attribute=protected_attribute, privileged_class=1)
    metric_name = "statistical_parity_ratio"
    prev_fairness_metric_score = fairness_metric.fairness_score(metric_name)

    # Create the BiasMitigation object to perform a bias mitigation
    bias_mitigation = BiasMitigation(ml_model=train_ml_model["ml_model"], data=train_ml_model["training_data"],
                                     target_attribute=train_ml_model["target_var"],
                                     protected_attribute=protected_attribute, privileged_class=1)

    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)

    if mitigation_method != "reweighing":
        mitigated_data = mitigation_res['data']
        X_train = mitigated_data.drop(columns=train_ml_model["target_var"])
        y_train = mitigated_data[train_ml_model["target_var"]]

        # ReTrain ml_model based on mitigated data
        train_ml_model["ml_model"].fit(X_train, y_train)
    else:
        train_ml_model["ml_model"].fit(train_ml_model["training_data"].drop(columns=train_ml_model["target_var"]),
                                       train_ml_model["training_data"][train_ml_model["target_var"]],
                                       sample_weight=mitigation_res['weights'])

    # Compute new Fairness score for "statistical_parity_ratio"
    fairness_metric = FairnessMetric(ml_model=train_ml_model["ml_model"], data=train_ml_model["testing_data"],
                                     target_attribute=train_ml_model["target_var"],
                                     protected_attribute=protected_attribute, privileged_class=1)
    new_fairness_metric_score = fairness_metric.fairness_score(metric_name)

    print(prev_fairness_metric_score[metric_name], new_fairness_metric_score[metric_name])
