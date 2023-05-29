import matplotlib
import pandas as pd
import numpy as np
import logging
import os
import copy
import math

# Ignore aif360 warnings
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Import library
from dalex.fairness import resample
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover

from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
from equalityml.threshold import binary_threshold_score, discrimination_threshold, DECISION_MAKER


class FAIR:
    """
    FAIR (Fairness Assessment and Inequality Reduction) empowers AI developers to assess fairness of their Machine
    Learning application and mitigate any observed bias in its application. It contains methods to assess fairness
    metrics as well as a set of bias algorithms for mitigating unfairness.

    Parameters
    ----------
    ml_model : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised.
    training_data : pd.DataFrame
        Training data in the form of a pd.DataFrame, which was used to train the machine learning model.
    target_variable : str
        Name of the target variable column in the training/testing data.
    protected_variable : str
        Name of the protected variable column in the training/testing data.
    privileged_class : float
        Subgroup that is suspected to have the most privilege.
        It needs to be a value present in `protected_variable` column.
    testing_data : pd.DataFrame, default=None
        Testing data in the form of a pd.DataFrame, which will be used to make predictions.
    features : list, default=None
        Data attributes used to train the machine learning model
    unprivileged_class : float, default=None
        Subgroup that is suspected to have the least privilege.
        It needs to be a value present in `protected_variable` column.
    favorable_label : float, default=1
        The favorable label value in the target variable which is considered favorable (i.e. "positive").
    unfavorable_label : float, default=0
        The favorable label value in the target variable which is considered unfavorable (i.e. "negative").
    threshold : float, default=0.5
        Discrimination threshold for predicting the favorable class.
    pred_class : list, default=None
        Predicted class labels for input 'testing_data' applied on machine learning model 'ml_model'.
    pred_prob : list, default=None
        Probability estimates for input 'testing_data' applied on machine learning model 'ml_model'.
    random_seed : int, default=None
        Used to set the numpy random seed in order to make numpy calls predictable.
    """

    def __init__(self,
                 ml_model,
                 training_data,
                 target_variable,
                 protected_variable,
                 privileged_class,
                 testing_data=None,
                 features=None,
                 unprivileged_class=None,
                 favorable_label=1,
                 unfavorable_label=0,
                 threshold=0.5,
                 pred_class=None,
                 pred_prob=None,
                 random_seed=None):

        super(FAIR, self).__init__()

        # Check ml_model and type of target
        if getattr(ml_model, "_estimator_type", None) != "classifier":
            raise TypeError("Model has to be a classifier")

        if type_of_target(training_data[target_variable]) != "binary":
            raise ValueError("Multiclass format is not supported")

        self.orig_ml_model = copy.deepcopy(ml_model)
        self.ml_model = copy.deepcopy(ml_model)
        self.training_data = training_data.copy()
        self.training_data.reset_index(drop=True, inplace=True)

        # Check if testing data is used to assess fairness metrics.
        if testing_data is None:
            self.testing_data = None
        else:
            self.testing_data = testing_data.copy()
            self.testing_data.reset_index(drop=True, inplace=True)

            if sorted(self.testing_data.columns) != sorted(self.training_data.columns):
                raise TypeError(f"Testing data structure is not the same as training data")

        # Check input arguments
        if target_variable not in self.training_data.columns:
            raise TypeError(f"Target variable {target_variable} is not part of training data")

        if protected_variable not in self.training_data.columns:
            raise TypeError(f"Protected variable {protected_variable} is not part of training data")

        if features is not None:
            if any([feature not in self.training_data.columns for feature in features]):
                raise TypeError(f"At least one feature of {features} are not part of training data")

        privileged_class = float(privileged_class)
        if privileged_class not in self.training_data[protected_variable].values:
            raise TypeError(f"Privileged class {privileged_class} shall be on data column {protected_variable}")

        if favorable_label not in self.training_data[target_variable] or unfavorable_label not in self.training_data[
            target_variable] or \
                sorted(list(set(self.training_data[target_variable]))) != sorted([favorable_label, unfavorable_label]):
            raise TypeError("Invalid value of favorable/unfavorable labels")

        # Features used to train the model
        if features is None:
            self.features = self.training_data.columns.drop(target_variable)
        else:
            self.features = features

        self.target_variable = target_variable
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_variable = protected_variable
        self.privileged_class = privileged_class
        self._threshold = threshold
        self._metric_name = None

        if unprivileged_class is None:
            _unprivileged_classes = list(set(self.training_data[protected_variable]).difference([privileged_class]))
            if len(_unprivileged_classes) != 1:
                raise ValueError("Use only binary classes")
            self.unprivileged_class = _unprivileged_classes[0]
        else:
            self.unprivileged_class = unprivileged_class

        self.pred_class = pred_class
        self.pred_prob = pred_prob
        self.unprivileged_groups = [{self.protected_variable: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_variable: [self.privileged_class]}]
        self.mitigated_testing_data = None
        self.mitigated_training_data = None

        # dalex sets matplotlib to use agg. Save current matplotlib backend status to revert it.
        self.matplotlib_backend = matplotlib.get_backend()

        # Set the numpy random seed
        np.random.seed(random_seed)

    @property
    def fairness_metrics_list(self):
        return ['treatment_equality_ratio',
                'treatment_equality_difference',
                'balance_positive_class',
                'balance_negative_class',
                'equal_opportunity_ratio',
                'accuracy_equality_ratio',
                'predictive_parity_ratio',
                'predictive_equality_ratio',
                'statistical_parity_ratio']

    @property
    def bias_mitigations_list(self):
        return ['disparate-impact-remover', 'resampling', 'resampling-uniform', 'resampling-preferential', 'reweighing',
                'correlation-remover']

    @property
    def map_bias_mitigation(self):
        return {'treatment_equality_ratio': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'treatment_equality_difference': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'balance_positive_class': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'balance_negative_class': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'equal_opportunity_ratio': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'accuracy_equality_ratio': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'predictive_parity_ratio': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'predictive_equality_ratio': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing'],
                'statistical_parity_ratio': ['disparate-impact-remover', 'resampling',
                                             'resampling-preferential', 'reweighing']}

    @property
    def threshold(self):
        """Get discrimination threshold"""
        return getattr(self, '_threshold', 0.5)

    @threshold.setter
    def threshold(self,
                  value):
        """Set discrimination threshold"""
        self._threshold = value

    def _predict_binary_prob(self,
                             ml_model,
                             data):
        """Predict binary probabilities estimates"""
        try:
            _pred_prob = ml_model.predict_proba(data[self.features])
            _pred_prob = _pred_prob[:, 1]  # keep probabilities for positive outcomes only
        except Exception:
            raise Exception("Not possible to predict estimates using the input machine learning model")

        return _pred_prob

    def _predict_binary_class(self,
                              ml_model,
                              data):
        """Predict binary classes"""
        try:
            _pred_class = np.asarray(list(map(lambda x: 1 if x > self._threshold else 0,
                                              ml_model.predict_proba(data[self.features])[:, -1])))
        except Exception:
            raise Exception("Not possible to predict classes using the input machine learning model")

        return _pred_class

    def _cr_removing_data(self,
                          data,
                          alpha=1.0,
                          cr=None):
        """
        Filters out sensitive correlations in a dataset using 'CorrelationRemover' function from fairlearn package.
        """

        # Getting correlation coefficient for mitigation_method 'correlation_remover'. The input alpha parameter is
        # used to control the level of filtering between the sensitive and non-sensitive features

        # remove the outcome variable and sensitive variable
        data_rm_columns = data.columns.drop([self.protected_variable, self.target_variable])

        if cr is None:
            cr = CorrelationRemover(sensitive_feature_ids=[self.protected_variable], alpha=alpha)
            cr = cr.fit(data.drop(columns=[self.target_variable]))

        data_std = cr.transform(data.drop(columns=[self.target_variable]))
        data_cr = pd.DataFrame(data_std, columns=data_rm_columns, index=data.index)

        # Concatenate data after correlation remover
        mitigated_data = pd.concat(
            [pd.DataFrame(data[self.target_variable]),
             pd.DataFrame(data[self.protected_variable]),
             data_cr], axis=1)

        # Keep the same columns order
        mitigated_data = mitigated_data[data.columns]
        return mitigated_data, cr

    def _disp_removing_data(self,
                            data,
                            repair_level=0.8):
        """
        Transforming input data using 'DisparateImpactRemover' from aif360 pacakge.
        """

        # putting data in specific standardize form required by the aif360 package
        aif_data = BinaryLabelDataset(favorable_label=self.favorable_label,
                                      unfavorable_label=self.unfavorable_label,
                                      df=data,
                                      label_names=[self.target_variable],
                                      protected_attribute_names=[self.protected_variable],
                                      privileged_protected_attributes=[[self.privileged_class]],
                                      unprivileged_protected_attributes=[[self.unprivileged_class]])

        DIR = DisparateImpactRemover(repair_level=repair_level)
        data_std = DIR.fit_transform(aif_data)
        mitigated_data = data_std.convert_to_dataframe()[0]

        # Keep the same columns order
        mitigated_data = mitigated_data[data.columns]
        return mitigated_data

    def _resampling_data(self,
                         data,
                         mitigation_method):
        """
        Resample the input data using 'resample' function from dalex package.
        """

        # Uniform resampling
        idx_resample = 0
        if (mitigation_method == "resampling-uniform") or (mitigation_method == "resampling"):
            idx_resample = resample(data[self.protected_variable],
                                    data[self.target_variable],
                                    type='uniform',
                                    verbose=False)
        # Preferential resampling
        elif mitigation_method == "resampling-preferential":
            _pred_prob = self._predict_binary_prob(self.orig_ml_model, data)
            idx_resample = resample(data[self.protected_variable],
                                    data[self.target_variable],
                                    type='preferential', verbose=False,
                                    probs=_pred_prob)

        mitigated_data = data.iloc[idx_resample, :]

        return mitigated_data

    def _reweighing_model(self,
                          data):
        """
        Obtain weights for model training using 'Reweighing' function from aif360 package.
        """
        # putting data in specific standardize form required by the aif360 package
        aif_data = BinaryLabelDataset(favorable_label=self.favorable_label,
                                      unfavorable_label=self.unfavorable_label,
                                      df=data,
                                      label_names=[self.target_variable],
                                      protected_attribute_names=[self.protected_variable],
                                      privileged_protected_attributes=[[self.privileged_class]],
                                      unprivileged_protected_attributes=[[self.unprivileged_class]])

        RW = Reweighing(unprivileged_groups=self.unprivileged_groups,
                        privileged_groups=self.privileged_groups)
        dataset_transf = RW.fit_transform(aif_data)

        return dataset_transf.instance_weights

    def bias_mitigation(self,
                        mitigation_method,
                        alpha=1.0,
                        repair_level=0.8):
        """
        Apply a mitigation method to the bias in Machine Learning application to make it more balanced.
        A set of mitigation method is available which mitigates the dataset or calculates mitigated weights to be used
        in the machine learning model.

        Parameters
        ----------
        mitigation_method : str
             Name of the mitigation method. Accepted values:
             "resampling"
             "resampling-preferential"
             "reweighing"
             "disparate-impact-remover"
             "correlation-remover"
        alpha : float, default=1.0
            parameter to control how much to filter, for alpha=1.0 we filter out
            all information while for alpha=0.0 we don't apply any.
        repair_level : float, default=0.8
            Repair amount. 0.0 is no repair while 1.0 is full repair.
        Returns
        ----------
        T : dictionary-like of shape
            Mitigated data/weights
            Notes:
            Output shouldn't have both 'training_data/testing_data' and 'weights' keys i.e. if method only changes data then
             key is 'training_data/testing_data' and if method changes machine learning model weights then 'weights' is in key

        """

        if mitigation_method not in ["resampling-uniform", "resampling", "resampling-preferential",
                                     "correlation-remover", "reweighing", "disparate-impact-remover"]:
            raise ValueError(f"Incorrect mitigation method {mitigation_method}")
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("'alpha' must be between 0.0 and 1.0.")
        if repair_level < 0.0 or repair_level > 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")

        mitigated_dataset = {}
        if "resampling" in mitigation_method:
            mitigated_training_data = self._resampling_data(self.training_data, mitigation_method)
            mitigated_dataset['training_data'] = mitigated_training_data
            self.mitigated_training_data = mitigated_training_data
            self.mitigated_testing_data = self.testing_data
        elif mitigation_method == "correlation-remover":
            mitigated_training_data, cr = self._cr_removing_data(self.training_data, alpha)
            mitigated_dataset['training_data'] = mitigated_training_data
            self.mitigated_training_data = mitigated_training_data

            if self.testing_data is not None:
                mitigated_testing_data, _ = self._cr_removing_data(self.testing_data, alpha, cr)
                mitigated_dataset['testing_data'] = mitigated_testing_data
                self.mitigated_testing_data = mitigated_testing_data
        elif mitigation_method == "reweighing":
            mitigated_weights = self._reweighing_model(self.training_data)
            mitigated_dataset['weights'] = mitigated_weights
            self.mitigated_training_data = self.training_data
            self.mitigated_testing_data = self.testing_data
        elif mitigation_method == "disparate-impact-remover":
            mitigated_training_data = self._disp_removing_data(self.training_data, repair_level)
            mitigated_dataset['training_data'] = mitigated_training_data
            self.mitigated_training_data = mitigated_training_data

            if self.testing_data is not None:
                mitigated_testing_data = self._disp_removing_data(self.testing_data, repair_level)
                mitigated_dataset['testing_data'] = mitigated_testing_data
                self.mitigated_testing_data = mitigated_testing_data

        # dalex sets matplotlib to use agg. Revert it to previous matplotlib backend
        matplotlib.use(self.matplotlib_backend)

        return mitigated_dataset

    def fairness_metric(self,
                        metric_name):
        """
        Fairness metric assessment based on privileged/unprivileged classes.

        Parameters
        ----------
        metric_name : str
            fairness metric name. Available options are:
               1. 'treatment_equality_ratio':
               2. 'treatment_equality_difference':
               3. 'balance_positive_class': Balance for positive class
               4. 'balance_negative_class': Balance for negative class
               5. 'equal_opportunity_ratio': Equal opportunity ratio
               6. 'accuracy_equality_ratio': Accuracy equality ratio
               7. 'predictive_parity_ratio':  Predictive parity ratio
               8. 'predictive_equality_ratio': Predictive equality ratio
               9. 'statistical_parity_ratio': Statistical parity ratio

        Returns
        ----------
        T : dictionary-like of shape
            Returns the fairness metric score for the input fairness metric name.
            Note that the returned score will be between 0.0 and 1.0. In cases where the fairness metric is > 1.0 it
            returns the equivalent 1/fairness metric score to allow for easy comparison.
        """

        metric_name = metric_name.lower()
        if metric_name not in self.fairness_metrics_list:
            raise ValueError(f"Provided invalid metric name {metric_name}")

        self._metric_name = metric_name

        fairness_metric = {}

        if self.mitigated_testing_data is not None:
            testing_data = self.mitigated_testing_data
        else:
            testing_data = self.testing_data if self.testing_data is not None else self.training_data

        # create a dataset according to structure required by the package AIF360
        aif_data = BinaryLabelDataset(favorable_label=self.favorable_label,
                                      unfavorable_label=self.unfavorable_label,
                                      df=testing_data,
                                      label_names=[self.target_variable],
                                      protected_attribute_names=[self.protected_variable],
                                      privileged_protected_attributes=[[self.privileged_class]],
                                      unprivileged_protected_attributes=[[self.unprivileged_class]])

        # fairness metric computation
        aif_data_pred = aif_data.copy()

        # Get predicted classes in case input argument 'pred_class' is None
        if self.pred_class is None:
            aif_data_pred.labels = self._predict_binary_class(self.ml_model, testing_data)
        else:
            aif_data_pred.labels = self.pred_class

        # Get probability estimates in case input argument 'pred_prob' is None
        if self.pred_prob is None:
            aif_data_pred.scores = self._predict_binary_prob(self.ml_model, testing_data)
        else:
            aif_data_pred.scores = self.pred_prob

        cm_pred_data = ClassificationMetric(aif_data,
                                            aif_data_pred,
                                            unprivileged_groups=self.unprivileged_groups,
                                            privileged_groups=self.privileged_groups)

        def try_division(x, y):
            if abs(y) < 1e-10:
                return 0
            else:
                return x / y

        def scale_division(x, y):
            result = try_division(x, y)
            if result > 1:
                return 1 / result
            else:
                return result

        # Treatment equality
        # Note that both treatment equality ratio and treatment equality difference are calculated
        privileged_ratio = try_division(cm_pred_data.num_false_negatives(True), cm_pred_data.num_false_positives(True))
        unprivileged_ratio = try_division(cm_pred_data.num_false_negatives(False),
                                          cm_pred_data.num_false_positives(False))

        # Get privileged_metrics and unprivileged_metrics to compute required ratios
        privileged_metrics = cm_pred_data.performance_measures(True)
        unprivileged_metrics = cm_pred_data.performance_measures(False)

        if metric_name == 'treatment_equality_ratio':
            fairness_metric = scale_division(unprivileged_ratio, privileged_ratio)
        if metric_name == 'treatment_equality_difference':
            fairness_metric = unprivileged_ratio - privileged_ratio
        if metric_name == 'balance_negative_class':
            fairness_metric = scale_division(unprivileged_metrics['GFPR'], privileged_metrics['GFPR'])
        if metric_name == 'balance_positive_class':
            fairness_metric = scale_division(unprivileged_metrics['GTPR'], privileged_metrics['GTPR'])
        if metric_name == 'equal_opportunity_ratio':
            fairness_metric = scale_division(unprivileged_metrics['TPR'], privileged_metrics['TPR'])
        if metric_name == 'accuracy_equality_ratio':
            fairness_metric = scale_division(unprivileged_metrics['ACC'], privileged_metrics['ACC'])
        if metric_name == 'predictive_parity_ratio':
            fairness_metric = scale_division(unprivileged_metrics['PPV'], privileged_metrics['PPV'])
        if metric_name == 'predictive_equality_ratio':
            fairness_metric = scale_division(unprivileged_metrics['FPR'], privileged_metrics['FPR'])
        if metric_name == 'statistical_parity_ratio':
            fairness_metric = scale_division(cm_pred_data.selection_rate(False), cm_pred_data.selection_rate(True))

        return fairness_metric

    def print_fairness_metrics(self):
        print("Available fairness metrics are: \n"
              "1. 'treatment_equality_ratio'\n"
              "2. 'treatment_equality_difference'\n"
              "3. 'balance_positive_class': Balance for positive class\n"
              "4. 'balance_negative_class': Balance for negative class\n"
              "5. 'equal_opportunity_ratio': Equal opportunity ratio\n"
              "6. 'accuracy_equality_ratio': Accuracy equality ratio\n"
              "7. 'predictive_parity_ratio':  Predictive parity ratio\n"
              "8. 'predictive_equality_ratio': Predictive equality ratio\n"
              "9. 'statistical_parity_ratio': Statistical parity ratio")

    def print_bias_mitigation_methods(self,
                                      metric_name=None):
        if metric_name == None:
            print("Available bias mitigation methods are: \n"
                  "1. 'resampling' or 'resampling-uniform'\n"
                  "2. 'resampling-preferential'\n"
                  "3. 'reweighing'\n"
                  "4. 'disparate-impact-remover'\n"
                  "5. 'correlation-remover'")
        else:
            metric_name = metric_name.lower()
            if metric_name not in self.fairness_metrics_list:
                raise ValueError(f"Provided invalid metric name {metric_name}")

            print(f"Available bias mitigation methods for '{metric_name}' are:")
            mitigation_methods = self.map_bias_mitigation[metric_name]
            for idx, mitigation_method in enumerate(mitigation_methods):
                print(f"{idx+1} '{mitigation_method}'")

    def update_classifier(self,
                          ml_model,
                          pred_class=None,
                          pred_prob=None):
        """
        Update the Machine Learning model classifier.
        After applying a bias mitigation method and retraining the ML model, it is necessary to update the model to
        reassess the fairness metric.

        Parameters
        ----------
        ml_model : a Scikit-Learn estimator
            A scikit-learn estimator that should be a classifier. If the model is
            not a classifier, an exception is raised.
        pred_class : list, default=None
            Predicted class labels for input 'testing_data' applied on machine learning model 'ml_model'.
        pred_prob : list, default=None
            Probability estimates for input 'testing_data' applied on machine learning model 'ml_model'.
        """
        if getattr(ml_model, "_estimator_type", None) != "classifier":
            raise TypeError("Model has to be a classifier")
        self.ml_model = copy.deepcopy(ml_model)

        if pred_class is not None and len(set(pred_class)) != 2:
            raise TypeError("Only binary classes are available")

        self.pred_class = pred_class

        if pred_prob is not None and any(i < 0 or i > 1 for i in pred_prob):
            raise TypeError("Probability estimates shall be between 0 and 1")

        self.pred_prob = pred_prob

    def model_mitigation(self,
                         mitigation_method,
                         alpha=1.0,
                         repair_level=0.8):
        """
        Apply a mitigation method to the Machine Learning model.
        Dependent on the input mitigation method, the mitigated dataset or weights computed by 'bias_mitigation' method
        are used to re-train the machine learning model which is then returned.

        Parameters
        ----------
        mitigation_method : str
             Name of the mitigation method. Accepted values:
             "resampling"
             "resampling-preferential"
             "reweighing"
             "disparate-impact-remover"
             "correlation-remover"
        alpha : float, default=1.0
            parameter to control how much to filter, for alpha=1.0 we filter out
            all information while for alpha=0.0 we don't apply any.
        repair_level : float, default=0.8
            Repair amount. 0.0 is no repair while 1.0 is full repair.
        Returns
        ----------
        model : a Scikit-Learn estimator
            Mitigated Machine Learning model
        """
        mitigation_result = self.bias_mitigation(mitigation_method=mitigation_method, alpha=alpha,
                                                 repair_level=repair_level)
        if mitigation_method == "reweighing":
            mitigated_weights = mitigation_result['weights']
            X_train = self.training_data[self.features]
            y_train = self.training_data[self.target_variable]

            # ReTrain Machine Learning model based on mitigated weights
            ml_model = clone(self.orig_ml_model)
            ml_model.fit(X_train, y_train, sample_weight=mitigated_weights)
        else:
            mitigated_data = mitigation_result['training_data']
            X_train = mitigated_data[self.features]
            y_train = mitigated_data[self.target_variable]

            # ReTrain Machine Learning model based on mitigated data
            ml_model = clone(self.orig_ml_model)
            ml_model.fit(X_train, y_train)

        self.update_classifier(ml_model)
        return ml_model

    def compare_mitigation_methods(self,
                                   scoring=None,
                                   utility_costs=None,
                                   metric_name=None,
                                   mitigation_methods=None,
                                   fairness_threshold=0.8,
                                   compute_discrimination_threshold=False,
                                   decision_maker=DECISION_MAKER,
                                   show=False,
                                   save_figure=False,
                                   **kwargs):
        """
        Compares different bias mitigation methods and evaluates the model performance using a scoring metric and a
        fairness metric.
        A DataFrame is returned containing the comparison results of model performance and fairness metric for each
        bias mitigation method.
        Parameters
        ----------
        scoring : str, callable. Default=None
            If None (default), uses 'accuracy' for sklearn classifiers
            If 'cost', uses utility_costs parameter to calculate the score
            If str, uses a sklearn scoring metric string identifier, for example {accuracy, f1, precision, recall, roc_auc}.
            If a callable object or function is provided, it has to agree with sklearn's signature 'scorer(estimator, X, y)'.
            Check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html for more information.
        utility_costs : list, default=None
            Utility costs for cost-sensitive learning. It has to be a 4 element list where the cost values correspond to the
            following cost sequence: [TP, FN, FP, TN]
        metric_name : str
            Fairness metric name. Available options are:
               1. 'treatment_equality_ratio':
               2. 'treatment_equality_difference':
               3. 'balance_positive_class': Balance for positive class
               4. 'balance_negative_class': Balance for negative class
               5. 'equal_opportunity_ratio': Equal opportunity ratio
               6. 'accuracy_equality_ratio': Accuracy equality ratio
               7. 'predictive_parity_ratio':  Predictive parity ratio
               8. 'predictive_equality_ratio': Predictive equality ratio
               9. 'statistical_parity_ratio': Statistical parity ratio
        mitigation_methods : list, default=None
            List of bias mitigation methods to mitigate the model.
            If a bias mitigation method is not available in FAIR class, it will be removed from the list.
            In case it's None, a default list will be computed based on the provided fairness metric.
        fairness_threshold : float, default=0.8
            Non-discrimination value.
        compute_discrimination_threshold : bool, default=True,
            Compute the best discrimination threshold for each mitigated model based on the input decision maker.
        decision_maker : tuple, default=('f1', 'max')
            The metric and decision to optimize the discrimination threshold. The metric shall be available in input metrics
            list and 3 different decisions can be used: 'max', 'min' or 'limit' with the following behaviour:
            - 'max' computes the threshold which maximizes the selected metric
            - 'min' computes the threshold which minimizes the selected metric
            - 'limit' requires an extra float parameter between 0 and 1. The optimal threshold is calculated when the
            selected metric reaches that limit.
        show : bool, default=False
            If True, shows the results of model performance and fairness metric for each bias mitigation method.
        save_figure : bool, default=False
            If True, save the figure.
        kwargs : dict
            Keyword arguments passed to the bias mitigation method.
        Returns
        ----------
        comparison_df : pd.DataFrame
            DataFrame containing the comparison results of model performance and fairness metric for each bias
            mitigation method.
        """

        if metric_name is None:
            if self._metric_name is None:
                raise AttributeError("Fairness metric is unknown.")
        else:
            metric_name = metric_name.lower()
            if metric_name not in self.fairness_metrics_list:
                raise ValueError(f"Provided invalid metric name {metric_name}")
            self._metric_name = metric_name

        if mitigation_methods is None:
            mitigation_methods = self.map_bias_mitigation[self._metric_name]
        else:
            mitigation_methods = list(mitigation_methods)
            for mitigation_method in mitigation_methods:
                if mitigation_method not in self.bias_mitigations_list:
                    print(f"Bias mitigation method {mitigation_method} is not available")
                    mitigation_methods.remove(mitigation_method)

        # Check scoring/scorer
        if scoring is None:
            if self.ml_model._estimator_type == "classifier":
                scoring = "accuracy"
            # elif self.ml_model._estimator_type == "regressor":
            #    scoring = "r2"
            else:
                raise AttributeError("Model must be a Classifier.")

        # Dataframe columns
        if compute_discrimination_threshold:
            df_columns = [str(scoring), self._metric_name, "discrimination_threshold"]
        else:
            df_columns = [str(scoring), self._metric_name]

        # Create the Dataframe for comparing models performance and fairness metric
        comparison_df = pd.DataFrame(columns=df_columns)
        testing_data = self.testing_data if self.testing_data is not None else self.training_data

        # Reference score and fairness metric
        score = binary_threshold_score(self.orig_ml_model,
                                       testing_data[self.features],
                                       testing_data[self.target_variable],
                                       scoring=scoring,
                                       threshold=self.threshold,
                                       utility_costs=utility_costs)
        fairness_metric = self.fairness_metric(self._metric_name)
        if compute_discrimination_threshold:
            comparison_df.loc['reference'] = [score, fairness_metric, self.threshold]
        else:
            comparison_df.loc['reference'] = [score, fairness_metric]

        # Iterate over mitigation methods list and re-evaluate score and fairness metric
        for mitigation_method in mitigation_methods:
            ml_model = self.model_mitigation(mitigation_method=mitigation_method, **kwargs)
            if self.mitigated_testing_data is not None:
                testing_data = self.mitigated_testing_data
            else:
                testing_data = self.testing_data if self.testing_data is not None else self.training_data

            # Compute discrimination threshold if it is True
            if compute_discrimination_threshold:
                dt = discrimination_threshold(ml_model,
                                              testing_data[self.features],
                                              testing_data[self.target_variable],
                                              fair_object=self,
                                              decision_maker=decision_maker,
                                              metrics=[decision_maker[0]],
                                              utility_costs=utility_costs,
                                              show=False,
                                              model_training=False)
            else:
                dt = self.threshold

            score = binary_threshold_score(ml_model,
                                           testing_data[self.features],
                                           testing_data[self.target_variable],
                                           scoring=scoring,
                                           threshold=dt,
                                           utility_costs=utility_costs)
            fairness_metric = self.fairness_metric(self._metric_name)
            if compute_discrimination_threshold:
                comparison_df.loc[mitigation_method] = [score, fairness_metric, dt]
            else:
                comparison_df.loc[mitigation_method] = [score, fairness_metric]

        if show:
            cmap = plt.get_cmap("tab10")
            score = comparison_df.loc['reference'][str(scoring)]
            fairness_metric = comparison_df.loc['reference'][str(self._metric_name)]
            fig = plt.gcf()
            ax = plt.gca()
            ax.plot(score, fairness_metric, marker='*', linestyle='', color=cmap(0), label='reference')
            for idx, mitigation_method in enumerate(mitigation_methods):
                score = comparison_df.loc[mitigation_method][str(scoring)]
                fairness_metric = comparison_df.loc[mitigation_method][str(self._metric_name)]
                ax.plot(score, fairness_metric, marker='o', linestyle='', color=cmap(1 + idx), label=mitigation_method)

            ax.axhline(
                fairness_threshold,
                ls="--",
                c="r",
                lw=1,
                label="fairness threshold",
            )

            ax.legend(frameon=True, loc="best")
            ax.set_title(f"{str(scoring)} vs {str(self._metric_name)}")
            ax.set_xlabel(str(scoring))
            ax.set_ylabel(str(self._metric_name))
            _x_min = math.floor(min(comparison_df[str(scoring)]) * 95) / 100.0
            _x_max = math.ceil(max(comparison_df[str(scoring)]) * 105) / 100.0
            ax.set_xlim(_x_min, _x_max)
            ax.set_ylim(0.0, 1.0)
            plt.show()

            if save_figure:
                filename = os.path.join(os.getcwd(), "compare_mitigation_methods")
                plt.savefig(filename)

        return comparison_df
