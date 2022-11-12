import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pytest

from src.FairPkg.bias_mitigation import BiasMitigation


def get_training_data():
    # Read training and testing data.
    target_var = "HOS"
    training_data = pd.read_csv("fairness_data/data_train.csv")
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]

    # Train Random Forest
    param_ml = {
        "n_estimators": 500,  # Number of trees in the forest
        "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
        "random_state": 0
    }
    mdl_clf = RandomForestClassifier(**param_ml)
    mdl_clf.fit(X_train, y_train)
    return mdl_clf, training_data, target_var


def test_bias_mitigation(get_training_data):

    # Create the BiasMitigation object to perform a bias mitigation
    protected_attribute = 'RACERETH'
    bias_mitigation = BiasMitigation(ml_model=mdl_clf, data=training_data, target_attribute=target_var,
                                     protected_attribute=protected_attribute, privileged_class=1)

    mitigation_method = "correlation-remover"
    # "resampling-preferential", "reweighing", "disparate-impact-remover", "correlation-remover"
    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)

    assert np.all(mitigation_res.protected_attribute == training_data.protected_attribute)
