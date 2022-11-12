import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pytest

from src.FairPkg.bias_mitigation import BiasMitigation

TESTS_PATH = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PATH = os.path.abspath(os.path.join(TESTS_PATH, os.pardir))


@pytest.fixture()
def train_default_ml():
    # Read training and testing data.
    target_var = "HOS"
    train_path = os.path.join(PACKAGE_PATH, 'src', 'FairPkg', 'fairness_data', 'data_train.csv')
    training_data = pd.read_csv(train_path)
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]

    # Train Random Forest
    param_ml = {
        "n_estimators": 500,  # Number of trees in the forest
        "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
        "random_state": 0
    }
    ml_model = RandomForestClassifier(**param_ml)
    ml_model.fit(X_train, y_train)
    return {"ml_model": ml_model, "training_data": training_data, "target_var": target_var}


def test_bias_mitigation(train_default_ml):

    # Create the BiasMitigation object to perform a bias mitigation
    protected_attribute = 'RACERETH'
    bias_mitigation = BiasMitigation(ml_model=train_default_ml["ml_model"], data=train_default_ml["training_data"],
                                     target_attribute=train_default_ml["target_var"],
                                     protected_attribute=protected_attribute, privileged_class=1)

    mitigation_method = "correlation-remover"
    # "resampling-preferential", "reweighing", "disparate-impact-remover", "correlation-remover"
    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)

    assert np.all(mitigation_res["data"][protected_attribute] == train_default_ml["training_data"][protected_attribute])
