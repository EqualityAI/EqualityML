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
from sklearn.linear_model import LogisticRegression
from fairml import FairnessMetric

# Train a machine learning model (for example LogisticRegression)
ml_model = LogisticRegression()
ml_model.fit(X_train, Y_train)

fairness_metric = FairnessMetric(ml_model=ml_model, data=testing_data,
                                 target_variable="Y",
                                 protected_variable="X1", privileged_class=1)
metric_name='statistical_parity_ratio'
fairness_metric_score = fairness_metric.fairness_score(metric_name)
```

In case the model is unfair in terms of checked fairness metric score, FairML provides a range of methods to try to
mitigate bias in Machine Learning models. For example, we can use 'correlation-remover' to perform mitigation on 
training dataset.

```python
from sklearn.linear_model import LogisticRegression
from fairml import BiasMitigation

# Train a machine learning model (for example LogisticRegression)
ml_model = LogisticRegression()
ml_model.fit(X_train, Y_train)

mitigation_method = "correlation-remover"
bias_mitigation = BiasMitigation(ml_model=ml_model, data=train_data,
                                 target_variable="Y",
                                 protected_variable="X1", privileged_class=1)

mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)
```

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind.

```sh
pip install -e '.[all]'
pytest tests
```

## Release History

* 0.0.1
    * Work in progress

## Contributing

Check out the [CONTRIBUTING](https://github.com/EqualityAI/fairml_py/CONTRIBUTING.md) file to learn how to contribute to our project, report bugs, or make feature requests.
