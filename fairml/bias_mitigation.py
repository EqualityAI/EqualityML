import pandas as pd
import numpy as np
import logging

# Ignore aif360 warnings
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Import library with bias mitigation methods
from dalex import Explainer
from dalex.fairness import resample
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover


class BiasMitigation:
    """
    Bias mitigation class. Apply a mitigation method to make a Dataset more balanced.

    Parameters
    ----------
    ml_model : object
        Trained Machine Learning model object (for example LogisticRegression object).
    data : pd.DataFrame
        Data in pd.DataFrame format to make it more balanced.
    target_variable : str
        Target column of the data with outputs / scores.
    protected_variable : str
        Data attribute for which fairness is desired.
    privileged_class : float
        Subgroup that is suspected to have the most privilege.
        It needs to be a value present in `protected_variable` column.
    unprivileged_class : float, default=None
        Subgroup that is suspected to have the least privilege.
        It needs to be a value present in `protected_variable` column.
    favorable_label : float, default=1
        Label value which is considered favorable (i.e. "positive").
    unfavorable_label : float, default=0
        Label value which is considered unfavorable (i.e. "negative").
    """

    def __init__(self,
                 ml_model,
                 data,
                 target_variable,
                 protected_variable,
                 privileged_class,
                 unprivileged_class=None,
                 favorable_label=1,
                 unfavorable_label=0):
        super(BiasMitigation, self).__init__()

        # Check input arguments
        if target_variable not in data.columns or protected_variable not in data.columns:
            raise TypeError(f"Target variable {target_variable} or protected variable {protected_variable} are not "
                            f"part of Data")

        if privileged_class not in data[protected_variable].values:
            raise TypeError(f"Privileged class {privileged_class} shall be on data column {protected_variable}")

        if favorable_label not in data[target_variable] or unfavorable_label not in data[target_variable] or \
                sorted(list(set(data[target_variable]))) != sorted([favorable_label, unfavorable_label]):
            raise TypeError("Invalid value of favorable/unfavorable labels")

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

        self.unprivileged_groups = [{self.protected_variable: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_variable: [self.privileged_class]}]

    def fit_transform(self, mitigation_method, alpha=1.0, repair_level=0.8):
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
            Mitigated data/weights and corresponding transforms/indexes
            Notes:
            Mitigated data and corresponding transform is stored as dictionary
            Output shouldn't have both 'data' and 'weights' keys i.e. if method only changes data then
             key is 'data' and if method changes machine learning model weights then 'weight' is in key

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
            mitigated_dataset = self._resampling_data(mitigation_method)
        elif mitigation_method == "correlation-remover":
            mitigated_dataset = self._cr_removing_data(alpha)
        elif mitigation_method == "reweighing":
            mitigated_dataset = self._reweighing_model()
        elif mitigation_method == "disparate-impact-remover":
            mitigated_dataset = self._disp_removing_data(repair_level)

        return mitigated_dataset

    def _resampling_data(self, mitigation_method):
        """
        Resample the input data using dalex module function.

        Parameters
        ----------
        mitigation_method : str
            Name of the mitigation method. Accepted values:
                "resampling",
                "resampling-uniform",
                "resampling-preferential"
        Returns
        ----------
        T : dictionary-like of shape
            Mitigated data and corresponding indexes
        """

        mitigated_dataset = {}

        # Uniform resampling
        idx_resample = 0
        if (mitigation_method == "resampling-uniform") or (mitigation_method == "resampling"):
            idx_resample = resample(self.data[self.protected_variable], self.data[self.target_variable],
                                    type='uniform',
                                    verbose=False)
        # preferential resampling
        elif mitigation_method == "resampling-preferential":
            exp = Explainer(self.ml_model, self.data[self.data.columns.drop(self.target_variable)].values,
                            self.data[self.target_variable].values, verbose=False)
            idx_resample = resample(self.data[self.protected_variable], self.data[self.target_variable],
                                    type='preferential', verbose=False,
                                    probs=exp.y_hat)

        mitigated_data = self.data.iloc[idx_resample, :]
        # mitigated data
        mitigated_dataset['data'] = mitigated_data
        # resample index
        mitigated_dataset['index'] = idx_resample

        return mitigated_dataset

    def _cr_removing_data(self, alpha=1.0):
        """
        Filters out sensitive correlations in a dataset using 'CorrelationRemover' function from fairlearn package.

        Parameters
        ----------
        alpha : float, default=1.0
            Parameter to control how much to filter, for alpha=1.0 we filter out
            all information while for alpha=0.0 we don't apply any.
        Returns
        ----------
        T : dictionary-like of shape
            Mitigated data and corresponding 'CorrelationRemover' object.
        """
        mitigated_dataset = {}

        # Getting correlation coefficient for mitigation_method 'correlation_remover'. The input alpha parameter is
        # used to control the level of filtering between the sensitive and non-sensitive features

        # remove the outcome variable and sensitive variable
        train_data_rm = self.data.drop([self.protected_variable, self.target_variable], axis=1)
        train_data_rm_cols = list(train_data_rm.columns)
        cr = CorrelationRemover(sensitive_feature_ids=[self.protected_variable], alpha=alpha)
        train_data_cr = cr.fit_transform(self.data.drop([self.target_variable], axis=1))
        train_data_cr = pd.DataFrame(train_data_cr, columns=train_data_rm_cols)

        # complete data after correlation remover
        train_data_mitigated = pd.concat(
            [pd.DataFrame(self.data[self.target_variable]), pd.DataFrame(self.data[self.protected_variable]),
             train_data_cr], axis=1)

        mitigated_dataset['data'] = train_data_mitigated
        # correlation transform as an object
        mitigated_dataset['transform'] = cr

        return mitigated_dataset

    def _reweighing_model(self):
        """
        Obtain weights for model training using 'Reweighing' function from aif360 package.

        Returns
        ----------
        T : dictionary-like of shape
            Balanced model weights and corresponding 'Reweighing' object.
        """

        mitigated_dataset = {}

        # putting data in specific standardize form required by the aif360 package
        train_data_std = StandardDataset(self.data,
                                         label_name=self.target_variable,
                                         favorable_classes=[self.favorable_label],
                                         protected_attribute_names=[self.protected_variable],
                                         privileged_classes=[[self.privileged_class]])

        RW = Reweighing(unprivileged_groups=self.unprivileged_groups, privileged_groups=self.privileged_groups)
        RW = RW.fit(train_data_std)
        # train data after reweighing
        train_data_std_m = RW.transform(train_data_std)
        # train_data_mitigated = train_data_std_m.convert_to_dataframe()[0]
        # train_data_std_m.features - mitigated data features (i.e. without target values)
        # train_data_std_m.labels.ravel() - mitigated data target values
        # train_data_std_m.instance_weights - mitigated data weights for machine learning model

        # mitigated data weights for machine learning model
        mitigated_dataset['weights'] = train_data_std_m.instance_weights
        # transform as an object
        mitigated_dataset['transform'] = RW

        return mitigated_dataset

    def _disp_removing_data(self, repair_level=0.8):
        """
        Transforming input data using 'DisparateImpactRemover' from aif360 pacakge.

        Parameters
        ----------
        repair_level : float, default=0.8
            Repair amount. 0.0 is no repair while 1.0 is full repair.
        Returns
        ----------
        T : dictionary-like of shape
            Mitigated data and corresponding 'DisparateImpactRemover' object
        """

        mitigated_dataset = {}

        # putting data in specific standardize form required by the aif360 package
        train_data_std = BinaryLabelDataset(favorable_label=self.favorable_label,
                                            unfavorable_label=self.unfavorable_label,
                                            df=self.data,
                                            label_names=[self.target_variable],
                                            protected_attribute_names=[self.protected_variable])

        DIR = DisparateImpactRemover(repair_level=repair_level)
        train_data_std = DIR.fit_transform(train_data_std)
        train_data_mitigated = train_data_std.convert_to_dataframe()[0]

        # train data after mitigation
        mitigated_dataset['data'] = train_data_mitigated
        #  transform as an object
        mitigated_dataset['transform'] = DIR

        return mitigated_dataset
