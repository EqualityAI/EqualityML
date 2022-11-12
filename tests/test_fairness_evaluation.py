import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import pytest

from src.FairPkg.fairness_evaluation import FairnessMetric

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(TESTS_PATH, os.pardir))


expected_metric_scores = {
    "treatment_equality_ratio": 0.91,
    "treatment_equality_difference": -0.93,
    "balance_negative_class": 0.77,
    "balance_positive_class": 0.91,
    "equal_opportunity_ratio": 0.53,
    "accuracy_equality_ratio": 1.07,
    "predictive_parity_ratio": 0.75,
    "predictive_equality_ratio": 0.65,
    "statistical_parity_ratio": 0.40,
}


@pytest.fixture()
def default_fairness_metric():
    # Read training and testing data.
    target_var = "HOS"
    train_path = os.path.join(PACKAGE_PATH, 'src', 'FairPkg', 'fairness_data', 'data_train.csv')
    training_data = pd.read_csv(train_path)
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    test_path = os.path.join(PACKAGE_PATH, 'src', 'FairPkg', 'fairness_data', 'data_test.csv')
    testing_data = pd.read_csv(test_path)

    # Train Random Forest
    param_ml = {
        "n_estimators": 500,  # Number of trees in the forest
        "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
        "random_state": 0
    }
    ml_model = RandomForestClassifier(**param_ml)
    ml_model.fit(X_train, y_train)

    fairness_metric = FairnessMetric(ml_model=ml_model, data=testing_data,
                                     target_attribute=target_var,
                                     protected_attribute='RACERETH', privileged_class=1)
    return fairness_metric

# =======================================================


def test_fairness_score(default_fairness_metric):
    # Compute Fairness score
    metric_name = "all"
    fairness_metric_score = default_fairness_metric.fairness_score(metric_name)

    assert round(fairness_metric_score['treatment_equality_ratio'], 2) == 0.91
    assert round(fairness_metric_score['treatment_equality_difference'], 2) == -0.93
    assert round(fairness_metric_score['balance_negative_class'], 2) == 0.77
    assert round(fairness_metric_score['balance_positive_class'], 2) == 0.91
    assert round(fairness_metric_score['equal_opportunity_ratio'], 2) == 0.53
    assert round(fairness_metric_score['accuracy_equality_ratio'], 2) == 1.07
    assert round(fairness_metric_score['predictive_parity_ratio'], 2) == 0.75
    assert round(fairness_metric_score['predictive_equality_ratio'], 2) == 0.65
    assert round(fairness_metric_score['statistical_parity_ratio'], 2) == 0.40


@pytest.mark.parametrize("metric_name", expected_metric_scores.keys())
def test_generated_metrics_smoke(default_fairness_metric, metric_name):
    fairness_metric_score = default_fairness_metric.fairness_score(metric_name)
    assert fairness_metric_score[metric_name] == pytest.approx(expected_metric_scores[metric_name])
