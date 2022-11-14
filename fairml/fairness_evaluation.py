import pandas as pd
import numpy as np
import logging
import warnings

# Import library with fairness metrics methods
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from dalex import Explainer

# Ignore warnings
warnings.filterwarnings("ignore")

# Add a custom logger for debugging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class FairnessMetric:
    """
    Fairness metric class to evaluate fairness of a Machine Learning application.

    Parameters
    ----------
    ml_model : object
        Trained Machine Learning model object (for example LogisticRegression object).
    data : pd.DataFrame
        Data in pd.DataFrame format which will be used to evaluate fairness.
    target_attribute : str
        Target column of the data with outputs / scores.
    protected_attribute : str
        Data attribute for which fairness is desired.
    privileged_class : float
        Subgroup that is suspected to have the most privilege.
        It needs to be a value present in `protected_attribute` vector.
    unprivileged_class : float, default=None
        Subgroup that is suspected to have the least privilege.
        It needs to be a value present in `protected_attribute` vector.
    favorable_label : float, default=1
        Label value which is considered favorable (i.e. "positive").
    unfavorable_label : float, default=0
        Label value which is considered unfavorable (i.e. "negative").
    pred_class : list, default=None
        Predicted class labels for input 'data' applied on input machine learning model object 'ml_model'.
    pred_prob : list, default=None
        Probability estimates for input 'data' applied on input machine learning model object 'ml_model'.
    """

    def __init__(self,
                 ml_model,
                 data,
                 target_attribute,
                 protected_attribute,
                 privileged_class,
                 unprivileged_class=None,
                 favorable_label=1,
                 unfavorable_label=0,
                 pred_class=None,
                 pred_prob=None):
        super(FairnessMetric, self).__init__()

        # Check input arguments
        # TODO check ml_model object
        assert all(np.issubdtype(dtype, np.number) for dtype in data.dtypes)
        assert target_attribute in data.columns
        assert protected_attribute in data.columns
        assert isinstance(privileged_class, (float, int))
        assert privileged_class in data[protected_attribute]
        assert isinstance(favorable_label, (float, int)) and isinstance(unfavorable_label, (float, int))
        assert favorable_label in data[target_attribute] and unfavorable_label in data[target_attribute]
        assert sorted(list(set(data[target_attribute]))) == sorted([favorable_label, unfavorable_label]), \
            "Incorrect favorable and/or unfavorable labels."

        self.ml_model = ml_model
        self.data = data
        self.target_attribute = target_attribute
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_attribute = protected_attribute
        self.privileged_class = privileged_class
        if unprivileged_class is None:
            _unprivileged_classes = list(set(data[protected_attribute]).difference([privileged_class]))
            assert len(_unprivileged_classes) == 1, "Only available to use one unprivileged class"
            self.unprivileged_class = _unprivileged_classes[0]
            logger.debug(f"Computed unprivileged class {self.unprivileged_class}")
        else:
            self.unprivileged_class = unprivileged_class

        # Compute predicted classes in case input argument 'pred_class' is None
        if pred_class is None:
            _features = self.data.drop(columns=target_attribute)
            self.pred_class = ml_model.predict(_features)
        else:
            self.pred_class = pred_class

        # Compute probability estimates in case input argument 'pred_prob' is None
        if pred_prob is None:
            _features = self.data.drop(columns=target_attribute)
            self.pred_prob = ml_model.predict_proba(_features)
            self.pred_prob = self.pred_prob[:, 1]  # keep probabilities for positive outcomes only
        else:
            self.pred_prob = pred_prob

        self.unprivileged_groups = [{self.protected_attribute: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_attribute: [self.privileged_class]}]

    def fairness_score(self, metric_name, cutoff=0.5):
        """
        Fairness metric evaluation using protected variable, privileged class, etc.

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
            Threshold value used for the machine learning classifier

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
        assert metric_name in aif360_list or metric_name in dalex_list or metric_name == 'all', \
            "Provided invalid metric name"

        # TODO check cutoff value
        logger.debug('Machine learning model threshold: {}'.format(cutoff))

        fairness_metric_score = {}

        # Evaluate fairness_metrics using aif360 module
        if (metric_name in aif360_list) or (metric_name == 'all'):

            # create a dataset according to structure required by the package AIF360
            data_input_std = BinaryLabelDataset(favorable_label=self.favorable_label,
                                                unfavorable_label=self.unfavorable_label,
                                                df=self.data,
                                                label_names=[self.target_attribute],
                                                protected_attribute_names=[self.protected_attribute],
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
            X_data = self.data.drop(columns=self.target_attribute)
            # target variable
            y_data = self.data[[self.target_attribute]]

            # create an explainer
            exp = Explainer(self.ml_model, X_data, y_data, verbose=False)
            # define protected variable and privileged group
            protected_vec = self.data[self.protected_attribute]

            fairness_object = exp.model_fairness(protected=protected_vec, privileged=str(self.privileged_class),
                                                 cutoff=cutoff)

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
