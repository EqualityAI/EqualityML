import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import pytest

from fairml.fairness_evaluation import FairnessMetric

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(TESTS_PATH, os.pardir))

expected_metric_scores = {
    "treatment_equality_ratio": 0.914,
    "treatment_equality_difference": -0.933,
    "balance_negative_class": 0.773,
    "balance_positive_class": 0.914,
    "equal_opportunity_ratio": 0.535,
    "accuracy_equality_ratio": 1.072,
    "predictive_parity_ratio": 0.753,
    "predictive_equality_ratio": 0.647,
    "statistical_parity_ratio": 0.400,
}


@pytest.fixture()
def train_ml_model():
    # TODO Review dataset to use for fairness testing
    # Read training and testing data.
    target_var = "HOS"
    train_path = os.path.join(PACKAGE_PATH, 'data', 'data_train.csv')
    training_data = pd.read_csv(train_path)
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    test_path = os.path.join(PACKAGE_PATH, 'data', 'data_test.csv')
    testing_data = pd.read_csv(test_path)

    # Train Random Forest
    param_ml = {
        "n_estimators": 500,  # Number of trees in the forest
        "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
        "random_state": 0
    }
    ml_model = RandomForestClassifier(**param_ml)
    ml_model.fit(X_train, y_train)

    return {"ml_model": ml_model, "testing_data": testing_data, "target_var": target_var}


# =======================================================


def test_fairness_score_smoke(train_ml_model):
    fairness_metric = FairnessMetric(ml_model=train_ml_model["ml_model"], data=train_ml_model["testing_data"],
                                     target_attribute=train_ml_model["target_var"],
                                     protected_attribute='RACERETH', privileged_class=1)
    # Compute Fairness score
    metric_name = "all"
    fairness_metric_score = fairness_metric.fairness_score(metric_name)

    res = {key: round(fairness_metric_score[key], 3) for key in fairness_metric_score}
    assert res == expected_metric_scores


@pytest.mark.parametrize("metric_name", expected_metric_scores.keys())
def test_fairness_scores(train_ml_model, metric_name):
    fairness_metric = FairnessMetric(ml_model=train_ml_model["ml_model"], data=train_ml_model["testing_data"],
                                     target_attribute=train_ml_model["target_var"],
                                     protected_attribute='RACERETH', privileged_class=1)
    fairness_metric_score = fairness_metric.fairness_score(metric_name)
    assert fairness_metric_score[metric_name] == pytest.approx(expected_metric_scores[metric_name], abs=1e-3)
