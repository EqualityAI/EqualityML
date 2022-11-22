import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest

from fairml import FairnessMetric

_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
_METRICS = ['treatment_equality_ratio', 'treatment_equality_difference', 'balance_positive_class',
            'balance_negative_class', 'equal_opportunity_ratio', 'accuracy_equality_ratio', 'predictive_parity_ratio',
            'predictive_equality_ratio', 'statistical_parity_ratio', 'all']


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("estimator", _ESTIMATORS)
def test_smoke(metric, estimator):
    np.random.seed(0)
    n = 100
    X0 = np.random.normal(size=n)
    X1 = np.random.choice([1, 2], size=n)
    Y = np.random.choice([0, 1], size=n)
    train_df = pd.DataFrame({"X0": X0, "X1": X1, "Y": Y})

    if estimator == SVC:
        _estimator = estimator(probability=True)
    else:
        _estimator = estimator()
    _estimator.fit(train_df.drop(columns="Y"), Y)

    fairness_metric = FairnessMetric(ml_model=_estimator, data=train_df,
                                     target_variable="Y",
                                     protected_variable="X1", privileged_class=1)

    fairness_metric_score = fairness_metric.fairness_score(metric)
    if metric == 'all':
        for metric_name in _METRICS:
            if metric_name == 'all':
                pass
            else:
                assert isinstance(fairness_metric_score[metric_name], float)
    else:
        assert isinstance(fairness_metric_score[metric], float)
