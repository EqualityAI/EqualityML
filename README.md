<!---
Copyright 2022 The EqualityAI Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# FairML

FairML provides a set of tools to handle the fairness of a Machine Learning application. With the increase use of 
Machine Learning solutions on a variety of critical real-world problems, it is imperative to guarantee that fair decisions are 
performed by those black-box Machine Learning models. FairML aims to simplify the complex process of evaluating the fairness of
ML models and dataset. It also provides a range of methods to mitigate bias in dataset and models.

Are you concerned that data and algorithmic biases lead to machine learning models that treat individuals unfavorably on 
the basis of characteristics such as race, gender or political orientation? Do you want to address fairness in machine 
learning but do not know where to start?

## Installation

FairML can be installed from [PyPI](https://pypi.org/project/fairml/) or using [conda-forge](https://anaconda.org/conda-forge/fairml).

```bash
pip install fairml

conda install -c conda-forge fairml
```

## Quick tour

Check out the example below to see how FairML can be used to evaluate the fairness of a Ml model and dataset.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fairml.FairPkg.fairness_evaluation import fairness_metrics

# Read training and testing data.
target_var = "HOS"
training_data = pd.read_csv("fairness_data/data_train.csv")
X_train = training_data.drop(columns=target_var)
y_train = training_data[target_var]
testing_data = pd.read_csv("fairness_data/data_test.csv")
X_test = testing_data.drop(columns=target_var)
y_test = testing_data[target_var]

# Train a Machine Learning Model
param_ml = {
    "n_estimators": 500,  # Number of trees in the forest
    "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
    "random_state": 0
}
mdl_clf = RandomForestClassifier(**param_ml)
mdl_clf.fit(X_train, y_train)

# Estimate prediction probability and predicted class of training data (Put empty dataframe for testing in order to estimate this)
pred_class = mdl_clf.predict(X_test)
pred_prob = mdl_clf.predict_proba(X_test)
pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

metric_name = "all"
param_fairness_metric = {
    'pred_prob_data': pred_prob,
    'pred_class_data': pred_class,
    'protected_var': 'RACERETH',
    'privileged_classes': [1],
    'unprivileged_classes': [2],
    'favorable_classes': [1],
    'unfavorable_classes': [0],
    'model': mdl_clf
}
fairness_metric_score = fairness_metrics(testing_data, target_var, metric_name, param_fairness_metric)
```

In case the model is unfair in terms of checked fairness metric score, FairML provides a range of methods to try to
mitigate bias in Machine Learning models. For example, we can use 'correlation-remover' to perform mitigation on 
training dataset.

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fairml.FairPkg.bias_mitigation import mitigation_methods

# Read training data.
target_var = "HOS"
training_data = pd.read_csv("fairness_data/data_train.csv")
X_train = training_data.drop(columns=target_var)
y_train = training_data[target_var]

# Train a Machine Learning Model
param_ml = {
    "n_estimators": 500,  # Number of trees in the forest
    "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
    "random_state": 0
}
mdl_clf = RandomForestClassifier(**param_ml)
mdl_clf.fit(X_train, y_train)

mitigation_method = "correlation-remover"
param_mitigation_method = {
    'protected_var': 'RACERETH',
    'privileged_classes': [1],
    'unprivileged_classes': [2],
    'favorable_classes': [1],
    'unfavorable_classes': [0],
    'model': mdl_clf
}
mitigation_res = mitigation_methods(training_data, target_var, mitigation_method, param_mitigation_method)
```

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind.

```sh
pip install requeriments.txt
pytest tests
```

## Release History

* 0.0.1
    * Work in progress

## Contributing

Check out the [CONTRIBUTING](https://github.com/EqualityAI/fairml_py/CONTRIBUTING.md) file to learn how to contribute to our project, report bugs, or make feature requests.
