import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pytest

from src.FairPkg.fairness_evaluation import FairnessMetric


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

def get_training_data():
    # Read training and testing data.
    target_var = "HOS"
    training_data = pd.read_csv("fairness_data/data_train.csv")
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    testing_data = pd.read_csv("fairness_data/data_test.csv")

    # Train Random Forest
    param_ml = {
        "n_estimators": 500,  # Number of trees in the forest
        "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
        "random_state": 0
    }
    mdl_clf = RandomForestClassifier(**param_ml)
    mdl_clf.fit(X_train, y_train)
    return mdl_clf, testing_data, target_var

# =======================================================

def test_fairness_score(get_training_data):

    # Compute Fairness score
    metric_name = "all"
    fairness_metric = FairnessMetric(ml_model=mdl_clf, data=testing_data, target_attribute=target_var,
                                     protected_attribute='RACERETH', privileged_class=1)
    fairness_metric_score = fairness_metric.fairness_score(metric_name)

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
def test_generated_metrics_smoke(metric_name):
    fairness_metric = FairnessMetric(ml_model=mdl_clf, data=testing_data, target_attribute=target_var,
                                     protected_attribute='RACERETH')
    fairness_metric_score = fairness_metric.fairness_score(metric_name)
    assert fairness_metric_score == pytest.approx(expected_metric_scores[metric_name])

