import numpy as np
import logging

# Ignore aif360 warnings
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Import library with fairness metrics methods
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from dalex import Explainer


class FairnessMetric:
    """
    Fairness metric class to evaluate fairness of a binary classification Machine Learning application.

    Parameters
    ----------
    ml_model : object
        Trained Machine Learning model object (for example LogisticRegression object).
    data : pd.DataFrame
        Data in pd.DataFrame format which will be used to evaluate fairness.
    target_variable : str
        Target column of the data with outputs / scores.
    protected_variable : str
        Data attribute for which fairness is desired.
    privileged_class : float
        Subgroup that is suspected to have the most privilege class.
        It needs to be a value present in `protected_variable` column.
    unprivileged_class : float, default=None
        Subgroup that is suspected to have the least privilege class.
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
                 data,
                 target_variable,
                 protected_variable,
                 privileged_class,
                 unprivileged_class=None,
                 favorable_label=1,
                 unfavorable_label=0,
                 pred_class=None,
                 pred_prob=None):
        super(FairnessMetric, self).__init__()

        # Check input arguments
        if not all(np.issubdtype(dtype, np.number) for dtype in data.dtypes):
            raise TypeError("Data shall be numeric")

        if target_variable not in data.columns or protected_variable not in data.columns:
            raise TypeError(f"Target variable {target_variable} or {protected_variable} are not part of Data")

        if not isinstance(privileged_class, (float, int)) or privileged_class not in data[protected_variable].values:
            raise TypeError(f"Invalid type/value of privileged class")

        if not isinstance(favorable_label, (float, int)) or not isinstance(unfavorable_label, (float, int)) or \
                favorable_label not in data[target_variable] or unfavorable_label not in data[target_variable] or \
                sorted(list(set(data[target_variable]))) != sorted([favorable_label, unfavorable_label]):
            raise TypeError("Invalid type/value of favorable/unfavorable labels")

        self.ml_model = ml_model
        self.data = data
        self.target_variable = target_variable
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_variable = protected_variable
        self.privileged_class = privileged_class
        if unprivileged_class is None:
            _unprivileged_classes = list(set(data[protected_variable]).difference([privileged_class]))
            if len(_unprivileged_classes) != 1:
                raise ValueError("Use only binary classes")
            self.unprivileged_class = _unprivileged_classes[0]
        else:
            self.unprivileged_class = unprivileged_class

        # Compute predicted classes in case input argument 'pred_class' is None
        if pred_class is None:
            try:
                _features = self.data.drop(columns=target_variable)
                self.pred_class = ml_model.predict(_features)
            except Exception:
                raise Exception("Not possible to predict classes using the input machine learning model")
        else:
            self.pred_class = pred_class

        # Compute probability estimates in case input argument 'pred_prob' is None
        if pred_prob is None:
            try:
                _features = self.data.drop(columns=target_variable)
                self.pred_prob = ml_model.predict_proba(_features)
                self.pred_prob = self.pred_prob[:, 1]  # keep probabilities for positive outcomes only
            except Exception:
                raise Exception("Not possible to predict estimates using the input machine learning model")
        else:
            self.pred_prob = pred_prob

        self.unprivileged_groups = [{self.protected_variable: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_variable: [self.privileged_class]}]

    def fairness_score(self, metric_name, cutoff=0.5):
        """
        Fairness metric evaluation based on privileged/unprivileged classes.

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

        #  PACKAGE: AIF360
        aif360_list = ['treatment_equality_ratio', 'treatment_equality_difference', 'balance_positive_class',
                       'balance_negative_class']
        #  PACKAGE: DALEX
        dalex_list = ['equal_opportunity_ratio', 'accuracy_equality_ratio', 'predictive_parity_ratio',
                      'predictive_equality_ratio', 'statistical_parity_ratio']

        metric_name = metric_name.lower()
        if metric_name not in aif360_list and metric_name not in dalex_list and metric_name != 'all':
            raise ValueError(f"Provided invalid metric name {metric_name}")

        if cutoff < 0.0:
            raise ValueError("Cutoff value shall be positive.")

        fairness_metric_score = {}

        # Evaluate fairness_metrics using aif360 module
        if (metric_name in aif360_list) or (metric_name == 'all'):

            # create a dataset according to structure required by the package AIF360
            data_input_std = BinaryLabelDataset(favorable_label=self.favorable_label,
                                                unfavorable_label=self.unfavorable_label,
                                                df=self.data,
                                                label_names=[self.target_variable],
                                                protected_attribute_names=[self.protected_variable],
                                                unprivileged_protected_attributes=self.unprivileged_groups)

            # fairness metric computation
            data_input_pred = data_input_std.copy()
            data_input_pred.scores = self.pred_prob  # predicted  probability
            data_input_pred.labels = self.pred_class  # predicted class
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
            X_data = self.data.drop(columns=self.target_variable)
            # target variable
            y_data = self.data[self.target_variable]

            # create an explainer
            exp = Explainer(self.ml_model, X_data, y_data, verbose=False)
            # define protected vector
            protected_vec = self.data[self.protected_variable]

            fairness_object = exp.model_fairness(protected=protected_vec, privileged=str(self.privileged_class),
                                                 cutoff=cutoff)

            df_fairness = fairness_object.result
            # Protected vector shall be binary therefore it's assumed fairness_result is also binary for each ratio.
            # Use ratio which is different from 1.0
            if (metric_name == 'equal_opportunity_ratio') or (metric_name == 'all'):
                try:
                    fairness_metric_score['equal_opportunity_ratio'] = df_fairness[df_fairness['TPR'] != 1.]['TPR'][0]
                except IndexError:
                    fairness_metric_score['equal_opportunity_ratio'] = float("NaN")
            if (metric_name == 'accuracy_equality_ratio') or (metric_name == 'all'):
                try:
                    fairness_metric_score['accuracy_equality_ratio'] = df_fairness[df_fairness['ACC'] != 1.]['ACC'][0]
                except IndexError:
                    fairness_metric_score['accuracy_equality_ratio'] = float("NaN")
            if (metric_name == 'predictive_parity_ratio') or (metric_name == 'all'):
                try:
                    fairness_metric_score['predictive_parity_ratio'] = df_fairness[df_fairness['PPV'] != 1.]['PPV'][0]
                except IndexError:
                    fairness_metric_score['predictive_parity_ratio'] = float("NaN")
            if (metric_name == 'predictive_equality_ratio') or (metric_name == 'all'):
                try:
                    fairness_metric_score['predictive_equality_ratio'] = df_fairness[df_fairness['FPR'] != 1.]['FPR'][0]
                except IndexError:
                    fairness_metric_score['predictive_equality_ratio'] = float("NaN")
            if (metric_name == 'statistical_parity_ratio') or (metric_name == 'all'):
                try:
                    fairness_metric_score['statistical_parity_ratio'] = df_fairness[df_fairness['STP'] != 1.]['STP'][0]
                except IndexError:
                    fairness_metric_score['statistical_parity_ratio'] = float("NaN")

        return fairness_metric_score
