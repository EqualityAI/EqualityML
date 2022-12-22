import pandas as pd
import logging

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


class FAIR:
    """
    FAIR (Fairness Assessment and Inequality Reduction) empowers AI developers to assess fairness of their Machine
    Learning application and mitigate any observed bias in its application. It contains methods to assess fairness
    metrics as well as bias mitigation algorithms.

    Parameters
    ----------
    ml_model : object
        Trained Machine Learning model object (for example LogisticRegression object).
    training_data : pd.DataFrame
        Data in pd.DataFrame format.
    target_variable : str
        Target column of the data with outputs / scores.
    protected_variable : str
        Data attribute for which fairness is desired.
    privileged_class : float
        Subgroup that is suspected to have the most privilege.
        It needs to be a value present in `protected_variable` column.
    testing_data : pd.DataFrame, default=None
        Data in pd.DataFrame format.
    features : list, default=None
        Data attributes used to train the machine learning model
    unprivileged_class : float, default=None
        Subgroup that is suspected to have the least privilege.
        It needs to be a value present in `protected_variable` column.
    favorable_label : float, default=1
        Label value which is considered favorable (i.e. "positive").
    unfavorable_label : float, default=0
        Label value which is considered unfavorable (i.e. "negative").
    pred_class : list, default=None
        Predicted class labels for input 'data' applied on machine learning model object 'ml_model'.
    pred_prob : list, default=None
        Probability estimates for input 'data' applied on machine learning model object 'ml_model'.
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
                 pred_class=None,
                 pred_prob=None):

        super(FAIR, self).__init__()

        # Check input arguments
        if target_variable not in training_data.columns:
            raise TypeError(f"Target variable {target_variable} is not part of training data")
        if protected_variable not in training_data.columns:
            raise TypeError(f"Protected variable {protected_variable} is not part of training data")
        if features is not None:
            if any([feature not in training_data.columns for feature in features]):
                raise TypeError(f"At least one feature of {features} are not part of training data")

        privileged_class = float(privileged_class)
        if privileged_class not in training_data[protected_variable].values:
            raise TypeError(f"Privileged class {privileged_class} shall be on data column {protected_variable}")

        if favorable_label not in training_data[target_variable] or unfavorable_label not in training_data[
            target_variable] or \
                sorted(list(set(training_data[target_variable]))) != sorted([favorable_label, unfavorable_label]):
            raise TypeError("Invalid value of favorable/unfavorable labels")

        # check if testing data is used to assess fairness metrics.
        if testing_data is None:
            self.testing_data = None
        else:
            self.testing_data = testing_data.copy()

        # Features used to train the model
        if features is None:
            self.features = training_data.columns.drop(target_variable)
        else:
            self.features = features

        self.ml_model = ml_model
        self.training_data = training_data.copy()
        self.target_variable = target_variable
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_variable = protected_variable
        self.privileged_class = privileged_class

        if unprivileged_class is None:
            _unprivileged_classes = list(set(training_data[protected_variable]).difference([privileged_class]))
            if len(_unprivileged_classes) != 1:
                raise ValueError("Use only binary classes")
            self.unprivileged_class = _unprivileged_classes[0]
        else:
            self.unprivileged_class = unprivileged_class

        self.pred_class = pred_class
        self.pred_prob = pred_prob
        self.unprivileged_groups = [{self.protected_variable: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_variable: [self.privileged_class]}]
        self.fairness_metrics = []
        self.metric_name = None
        self.cutoff = None
        self.mitigated_testing_data = None
        self.mitigated_training_data = None

    def _get_binary_prob(self, data):

        try:
            _pred_prob = self.ml_model.predict_proba(data[self.features])
            _pred_prob = _pred_prob[:, 1]  # keep probabilities for positive outcomes only
        except Exception:
            raise Exception("Not possible to predict estimates using the input machine learning model")

        return _pred_prob

    def _get_binary_class(self, data):

        try:
            _pred_class = self.ml_model.predict(data[self.features])
        except Exception:
            raise Exception("Not possible to predict classes using the input machine learning model")

        return _pred_class

    def _get_fairness_metric(self, df_fairness, metric):

        try:
            _fairness_metric = df_fairness[df_fairness[metric] != 1.][metric][0]
        except IndexError:
            _fairness_metric = float("NaN")

        return _fairness_metric

    def _cr_removing_data(self, data, alpha=1.0):
        """
        Filters out sensitive correlations in a dataset using 'CorrelationRemover' function from fairlearn package.
        """

        # Getting correlation coefficient for mitigation_method 'correlation_remover'. The input alpha parameter is
        # used to control the level of filtering between the sensitive and non-sensitive features

        # remove the outcome variable and sensitive variable
        data_rm_columns = data.columns.drop([self.protected_variable, self.target_variable])

        if not hasattr(self, 'cr'):
            self.cr = CorrelationRemover(sensitive_feature_ids=[self.protected_variable], alpha=alpha)
            data_std = self.cr.fit_transform(data.drop(columns=[self.target_variable]))
        else:
            data_std = self.cr.transform(data.drop(columns=[self.target_variable]))
        train_data_cr = pd.DataFrame(data_std, columns=data_rm_columns)

        # Concatenate data after correlation remover
        mitigated_data = pd.concat(
            [pd.DataFrame(data[self.target_variable]),
             pd.DataFrame(data[self.protected_variable]),
             train_data_cr], axis=1)

        # Keep the same columns order
        mitigated_data = mitigated_data[data.columns]
        return mitigated_data

    def _disp_removing_data(self, data, repair_level=0.8):
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

    def _resampling_data(self, data, mitigation_method):
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
            _pred_prob = self._get_binary_prob(data)
            idx_resample = resample(data[self.protected_variable],
                                    data[self.target_variable],
                                    type='preferential', verbose=False,
                                    probs=_pred_prob)

        mitigated_data = data.iloc[idx_resample, :]

        return mitigated_data

    def _reweighing_model(self, data):
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
        dataset_transf_train = RW.fit_transform(aif_data)

        return dataset_transf_train.instance_weights

    def bias_mitigation(self, mitigation_method, alpha=1.0, repair_level=0.8):
        """
        Apply a mitigation method to data to make it more balanced.

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
            Output shouldn't have both 'training_data' and 'weights' keys i.e. if method only changes data then
             key is 'data' and if method changes machine learning model weights then 'weights' is in key

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
        elif mitigation_method == "correlation-remover":
            mitigated_training_data = self._cr_removing_data(self.training_data, alpha)
            mitigated_dataset['training_data'] = mitigated_training_data
            self.mitigated_training_data = mitigated_training_data

            if self.testing_data is not None:
                mitigated_testing_data = self._cr_removing_data(self.testing_data, alpha)
                mitigated_dataset['testing_data'] = mitigated_testing_data
                self.mitigated_testing_data = mitigated_testing_data
        elif mitigation_method == "reweighing":
            mitigated_weights = self._reweighing_model(self.training_data)
            mitigated_dataset['weights'] = mitigated_weights
        elif mitigation_method == "disparate-impact-remover":
            mitigated_training_data = self._disp_removing_data(self.training_data, repair_level)
            mitigated_dataset['training_data'] = mitigated_training_data
            self.mitigated_training_data = mitigated_training_data

            if self.testing_data is not None:
                mitigated_testing_data = self._disp_removing_data(self.testing_data, repair_level)
                mitigated_dataset['testing_data'] = mitigated_testing_data
                self.mitigated_testing_data = mitigated_testing_data

        return mitigated_dataset

    def fairness_metric(self, metric_name, cutoff=0.5):
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
               10. 'all'
        cutoff : float, default=0.5
            Cutoff for predictions in classification models. Needed for measures like recall, precision, acc, f1

        Returns
        ----------
        T : dictionary-like of shape
            Returns the fairness metric score for the input fairness metric name.
        """

        metrics_list = ['treatment_equality_ratio', 'treatment_equality_difference', 'balance_positive_class',
                        'balance_negative_class', 'equal_opportunity_ratio', 'accuracy_equality_ratio',
                        'predictive_parity_ratio', 'predictive_equality_ratio', 'statistical_parity_ratio']

        metric_name = metric_name.lower()
        if metric_name not in metrics_list and metric_name != 'all':
            raise ValueError(f"Provided invalid metric name {metric_name}")

        if cutoff < 0.0:
            raise ValueError("Cutoff value shall be positive.")

        self.metric_name = metric_name
        self.cutoff = cutoff

        fairness_metric = {}

        if self.mitigated_testing_data is None:
            if self.testing_data is None:
                testing_data = self.training_data
            else:
                testing_data = self.testing_data
        else:
            testing_data = self.mitigated_testing_data

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
            aif_data_pred.labels = self._get_binary_class(testing_data)
        else:
            aif_data_pred.labels = self.pred_class

        # Get probability estimates in case input argument 'pred_prob' is None
        if self.pred_prob is None:
            aif_data_pred.scores = self._get_binary_prob(testing_data)
        else:
            aif_data_pred.scores = self.pred_prob

        cm_pred_data = ClassificationMetric(aif_data,
                                            aif_data_pred,
                                            unprivileged_groups=self.unprivileged_groups,
                                            privileged_groups=self.privileged_groups)
        # Treatment equality
        # Note that both treatment equality ratio and treatment equality difference are calculated
        privileged_ratio = cm_pred_data.num_false_negatives(True) / cm_pred_data.num_false_positives(True)
        unprivileged_ratio = cm_pred_data.num_false_negatives(False) / cm_pred_data.num_false_positives(False)

        # Get privileged_metrics and unprivileged_metrics to compute required ratios
        privileged_metrics = cm_pred_data.performance_measures(True)
        unprivileged_metrics = cm_pred_data.performance_measures(False)

        if (metric_name == 'treatment_equality_ratio') or (metric_name == 'all'):
            fairness_metric['treatment_equality_ratio'] = unprivileged_ratio / privileged_ratio
        if (metric_name == 'treatment_equality_difference') or (metric_name == 'all'):
            fairness_metric['treatment_equality_difference'] = unprivileged_ratio - privileged_ratio
        if (metric_name == 'balance_negative_class') or (metric_name == 'all'):
            fairness_metric['balance_negative_class'] = unprivileged_metrics['GFPR'] / privileged_metrics['GFPR']
        if (metric_name == 'balance_positive_class') or (metric_name == 'all'):
            fairness_metric['balance_positive_class'] = unprivileged_metrics['GTPR'] / privileged_metrics['GTPR']
        if (metric_name == 'equal_opportunity_ratio') or (metric_name == 'all'):
            fairness_metric['equal_opportunity_ratio'] = unprivileged_metrics['TPR'] / privileged_metrics['TPR']
        if (metric_name == 'accuracy_equality_ratio') or (metric_name == 'all'):
            fairness_metric['accuracy_equality_ratio'] = unprivileged_metrics['ACC'] / privileged_metrics['ACC']
        if (metric_name == 'predictive_parity_ratio') or (metric_name == 'all'):
            fairness_metric['predictive_parity_ratio'] = unprivileged_metrics['PPV'] / privileged_metrics['PPV']
        if (metric_name == 'predictive_equality_ratio') or (metric_name == 'all'):
            fairness_metric['predictive_equality_ratio'] = unprivileged_metrics['FPR'] / privileged_metrics['FPR']
        if (metric_name == 'statistical_parity_ratio') or (metric_name == 'all'):
            fairness_metric['statistical_parity_ratio'] = cm_pred_data.selection_rate(False) / \
                                                          cm_pred_data.selection_rate(True)

        self.fairness_metrics.append(fairness_metric)

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

    def print_bias_mitigation_methods(self):
        print("Available bias mitigation methods are: \n"
              "1. 'resampling' or 'resampling-uniform'\n"
              "2. 'resampling-preferential'\n"
              "3. 'reweighing'\n"
              "4. 'disparate-impact-remover'\n"
              "5. 'correlation-remover'")

    def update_classifier(self, ml_model, pred_class=None, pred_prob=None):
        """
        Update Machine Learning model classifier typically after applying a bias mitigation method.

        Parameters
        ----------
        ml_model : object
            Trained Machine Learning model object (for example LogisticRegression object).
        pred_class : list, default=None
            Predicted class labels for input 'data' applied on machine learning model object 'ml_model'.
        pred_prob : list, default=None
            Probability estimates for input 'data' applied on machine learning model object 'ml_model'.
        """
        self.ml_model = ml_model
        self.pred_class = pred_class
        self.pred_prob = pred_prob
