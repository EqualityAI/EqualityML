"""
FAIRNESS EVALUATION METRICS
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from dalex import Explainer
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class FairnessMetric:
    """
    Fairness metric evaluation using protected variable, privileged class, etc.
        """

    def __init__(self,
                 ml_model: object,
                 data: pd.DataFrame,
                 target_col: str,
                 favorable_label: int,
                 protected_col: str,
                 privileged_class: int,
                 unprivileged_class=None,
                 unfavorable_label=None,
                 pred_prob_data=None,
                 pred_class_data=None,
                 cutoff=0.5):
        """
        Apply a mitigation method to data to make in more balanced.

        Args:
            ml_model (object):
                Target variable in the input data
            data (pd.DataFrame):
                Input data in dictionary format to evaluate fairness
            target_col (str):
                Target variable in the input data
            favorable_label (int):
                Target variable in the input data
            protected_col (str):
                Target variable in the input data
            unfavorable_label (int):
                Target variable in the input data
            pred_prob_data (int):
                Target variable in the input data
            pred_class_data (int):
                Target variable in the input data
            cutoff (float):
                Threshold value used for the machine learning classifier
        """
        super(FairnessMetric, self).__init__()

        assert all(np.issubdtype(dtype, np.number) for dtype in data.dtypes)
        assert target_col in data.columns
        assert isinstance(favorable_label, (float, int))

        self.ml_model = ml_model
        self.data = data
        self.target_col = target_col
        self.favorable_label = favorable_label

        if unfavorable_label is None:
            _unfavorable_labels = list(set(self.data[target_col]))
            _unfavorable_labels.remove(self.favorable_label)
            self.unfavorable_label = _unfavorable_labels[0]  # just use one unfavorable label
            logger.debug(f"Computed unfavorable label {self.unfavorable_label}")
        else:
            self.unfavorable_label = unfavorable_label

        self.protected_col = protected_col
        self.privileged_class = privileged_class
        if unprivileged_class is None:
            _unprivileged_classes = list(set(self.data[protected_col]))
            _unprivileged_classes.remove(self.privileged_class)
            self.unprivileged_class = _unprivileged_classes[0]  # just use one unprivileged class
            logger.debug(f"Computed unprivileged class {self.unprivileged_class}")
        else:
            self.unprivileged_class = unprivileged_class
        self.cutoff = cutoff

        if pred_class_data is None:
            _features = self.data.drop(columns=target_col)
            self.pred_class_data = mdl_clf.predict(_features)
        else:
            self.pred_class_data = pred_class_data

        if pred_prob_data is None:
            _features = self.data.drop(columns=target_col)
            self.pred_prob_data = mdl_clf.predict_proba(_features)
            self.pred_prob_data = self.pred_prob_data[:, 1]  # keep probabilities for positive outcomes only
        else:
            self.pred_prob_data = pred_prob_data

        self.unprivileged_groups = [{self.protected_col: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_col: [self.privileged_class]}]

    def check_fairness(self, metric_name: str) -> dict:
        """
        Fairness metric evaluation using protected variable, privileged class, etc.

        Args:
            metric_name (str): fairness metric name. Available options are:
               1. 'treatment_equality_ratio':
               2. 'treatment_equality_diff':
               3. 'balance_positive_class': Balance for positive class
               4. 'balance_negative_class': Balance for negative class
               5. 'equal_opportunity_ratio': Equal opportunity ratio
               6. 'accuracy_equality_ratio': Accuracy equality ratio
               7. 'predictive_parity_ratio':  Predictive parity ratio
               8. 'predictive_equality_ratio': Predictive equality ratio
               9. 'statistical_parity_ratio': Statistical parity ratio
               10. 'all'

        Returns:
            dict: fairness metric value
        """

        #  PACKAGE: AIF360
        aif360_list = ['treatment_equality_ratio', 'treatment_equality_diff', 'balance_positive_class',
                       'balance_negative_class']
        #  PACKAGE: DALEX
        dalex_list = ['equal_opportunity_ratio', 'accuracy_equality_ratio', 'predictive_parity_ratio',
                      'predictive_equality_ratio', 'statistical_parity_ratio']

        metric_name = metric_name.lower()
        assert metric_name in aif360_list or metric_name in dalex_list or metric_name == 'all', \
            "Provided invalid metric name"

        fairness_metric_score = {}

        # Evaluate fairness_metrics using aif360 module
        if (metric_name in aif360_list) or (metric_name == 'all'):

            # create a dataset according to structure required by the package AIF360
            data_input_std = BinaryLabelDataset(favorable_label=self.favorable_label,
                                                unfavorable_label=self.unfavorable_label,
                                                df=self.data,
                                                label_names=[self.target_col],
                                                protected_attribute_names=[self.protected_col],
                                                unprivileged_protected_attributes=self.unprivileged_groups)

            # fairness metric computation
            data_input_pred = data_input_std.copy()
            data_input_pred.scores = self.pred_prob_data  # predicted  probability
            data_input_pred.labels = self.pred_class_data  # predicted class
            cm_pred_data = ClassificationMetric(data_input_std,
                                                data_input_pred,
                                                unprivileged_groups=self.unprivileged_groups,
                                                privileged_groups=self.privileged_groups)
            # Treatment equality
            # Note that both treatment equality ratio and treatment equality difference are calculated
            prev_rat = cm_pred_data.num_false_negatives(True) / cm_pred_data.num_false_positives(True)
            unprev_rat = cm_pred_data.num_false_negatives(False) / cm_pred_data.num_false_positives(False)
            treatment_equality_ratio = unprev_rat / prev_rat
            treatment_equality_diff = unprev_rat - prev_rat

            if (metric_name == 'treatment_equality_ratio') or (metric_name == 'all'):
                fairness_metric_score['treatment_equality_ratio'] = treatment_equality_ratio
            if (metric_name == 'treatment_equality_difference') or (metric_name == 'all'):
                fairness_metric_score['treatment_equality_difference'] = treatment_equality_diff
            if (metric_name == 'balance_negative_class') or (metric_name == 'all'):
                fairness_metric_score['balance_negative_class'] = cm_pred_data.ratio(
                    cm_pred_data.generalized_false_positive_rate)
            if (metric_name == 'balance_positive_class') or (metric_name == 'all'):
                fairness_metric_score['balance_positive_class'] = cm_pred_data.ratio(
                    cm_pred_data.generalized_true_positive_rate)

        # Evaluate fairness_metrics using dalex module
        if (metric_name in dalex_list) or (metric_name == 'all'):
            # features
            X_data = self.data.drop(columns=self.target_col)
            # target variable
            y_data = self.data[[self.target_col]]

            # create an explainer
            exp = Explainer(self.ml_model, X_data, y_data, verbose=False)
            # define protected variable and privileged group
            protected_vec = self.data[self.protected_col]

            logger.debug('Machine learning model threshold: {}'.format(self.cutoff))
            fairness_object = exp.model_fairness(protected=protected_vec, privileged=str(self.privileged_class),
                                                 cutoff=self.cutoff)

            fairness_result = fairness_object.result
            if (metric_name == 'equal_opportunity_ratio') or (metric_name == 'all'):
                # NOTE: check index location for different values of privileged class
                # TEST: Does location is dependent on the value of the privileged class?
                fairness_metric_score['equal_opportunity_ratio'] = fairness_result['TPR'][1]
            if (metric_name == 'accuracy_equality_ratio') or (metric_name == 'all'):
                fairness_metric_score['accuracy_equality_ratio'] = fairness_result['ACC'][1]
            if (metric_name == 'predictive_parity_ratio') or (metric_name == 'all'):
                fairness_metric_score['predictive_parity_ratio'] = fairness_result['PPV'][1]
            if (metric_name == 'predictive_equality_ratio') or (metric_name == 'all'):
                fairness_metric_score['predictive_equality_ratio'] = fairness_result['FPR'][1]
            if (metric_name == 'statistical_parity_ratio') or (metric_name == 'all'):
                fairness_metric_score['statistical_parity_ratio'] = fairness_result['STP'][1]

        return fairness_metric_score


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    # First train a Machine Learning model with the training data

    # Read training and testing data.
    target_var = "HOS"
    training_data = pd.read_csv("fairness_data/data_train.csv")
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    testing_data = pd.read_csv("fairness_data/data_test.csv")
    X_test = testing_data.drop(columns=target_var)
    y_test = testing_data[target_var]

    # Train Random Forest
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

    auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    print(f"AUC = {auc}")

    accuracy = accuracy_score(y_test, pred_class)  # classification accuracy
    print(f"Accuracy = {accuracy}")

    metric_name = "all"
    fairness_metric = FairnessMetric(ml_model=mdl_clf, data=testing_data, target_col=target_var, favorable_label=1,
                                     protected_col='RACERETH', privileged_class=1)
    fairness_metric_score = fairness_metric.check_fairness(metric_name)
    print(fairness_metric_score)
