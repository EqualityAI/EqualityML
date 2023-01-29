import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pytest

from equalityml.fair import FAIR
from equalityml.threshold import discrimination_threshold

_DECISION_THRESHOLD = [['f1', 'max'], ['recall', 'max'], ['precision', 'max'], ['queue_rate', 'limit', '0.1'], ['something']]

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


@pytest.mark.parametrize("decision_threshold", _DECISION_THRESHOLD)
def test_discrimination_threshold(dataset, decision_threshold):
    np.random.seed(0)

    # Machine learning model
    model = LogisticRegression()

    # FAIR object
    fair_object = FAIR(ml_model=model, training_data=dataset["training_data"],
                       target_variable=dataset["target_variable"],
                       protected_variable=dataset["protected_variable"], privileged_class=1)

    # Compute the best discrimination_threshold
    dt = discrimination_threshold(model,
                                  dataset["training_data"],
                                  dataset["target_variable"],
                                  fair_object=fair_object,
                                  fairness_metric_name='statistical_parity_ratio',
                                  decision_threshold=decision_threshold)

    assert 0 <= dt <= 1

