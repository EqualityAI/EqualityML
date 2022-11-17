import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest

from fairml import BiasMitigation

_ESTIMATORS = [LogisticRegression, SVC, DecisionTreeClassifier, RandomForestClassifier]
_MITIGATION_METHODS = ["resampling", "resampling-preferential", "reweighing", "disparate-impact-remover",
                       "correlation-remover"]


@pytest.mark.parametrize("mitigation_method", _MITIGATION_METHODS)
@pytest.mark.parametrize("estimator", _ESTIMATORS)
def test_smoke(mitigation_method, estimator):
    np.random.seed(0)
    n = 100
    X0 = np.random.normal(size=n)
    X1 = np.random.choice([1, 2], size=n)
    Y = np.random.choice([0, 1], size=n)
    df = pd.DataFrame({"X0": X0, "X1": X1, "Y": Y})

    _estimator = estimator()
    _estimator.fit(df.drop(columns="Y"), Y)

    bias_mitigation = BiasMitigation(ml_model=_estimator, data=df,
                                     target_variable="Y",
                                     protected_variable="X1", privileged_class=1)

    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)
    assert isinstance(mitigation_res, dict)
