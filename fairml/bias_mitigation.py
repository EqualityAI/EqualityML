import pandas as pd
import numpy as np
import logging
import warnings

# Import library with bias mitigation methods
from dalex import Explainer
from dalex.fairness import resample
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover

# Ignore warnings
warnings.filterwarnings("ignore")


class BiasMitigation:
    """
    Bias mitigation class. Apply a mitigation method to make a Dataset more balanced.

    Parameters
    ----------
    ml_model : object
        Trained Machine Learning model object (for example LogisticRegression object).
    data : pd.DataFrame
        Data in pd.DataFrame format to make it more balanced.
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
    """

    def __init__(self,
                 ml_model,
                 data,
                 target_attribute,
                 protected_attribute,
                 privileged_class,
                 unprivileged_class=None,
                 favorable_label=1,
                 unfavorable_label=0):
        super(BiasMitigation, self).__init__()

        # Check input arguments
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
            self.unprivileged_class = _unprivileged_classes[0]  # just use one unprivileged class
        else:
            self.unprivileged_class = unprivileged_class

        self.unprivileged_groups = [{self.protected_attribute: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_attribute: [self.privileged_class]}]

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
            'mitigation_output' method key shouldn't have both 'data' and 'weights' i.e. if method only changes data then
             key is 'data' and if method changes machine learning model weights then 'weight' is in key

        """

        assert mitigation_method in ["resampling-uniform", "resampling", "resampling-preferential",
                                     "correlation-remover",
                                     "reweighing", "disparate-impact-remover"], "Incorrect mitigation method."
        assert 0.0 <= alpha <= 1.0, "'alpha' must be between 0.0 and 1.0."
        assert 0.0 <= repair_level <= 1.0, "'repair_level' must be between 0.0 and 1.0."

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
        Resample the input data using dalex module functions.

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
            idx_resample = resample(self.data[self.protected_attribute], self.data[self.target_attribute],
                                    type='uniform',
                                    verbose=False)
        # preferential resampling
        elif mitigation_method == "resampling-preferential":
            exp = Explainer(self.ml_model, self.data[self.data.columns.drop(self.target_attribute)].values,
                            self.data[self.target_attribute].values, verbose=False)
            idx_resample = resample(self.data[self.protected_attribute], self.data[self.target_attribute],
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

        # Getting correlation coefficient for mitigation_method 'correlation_remover'
        # The alpha parameter is used to control the level of filtering between the sensitive and non-sensitive features

        # remove the outcome variable and sensitive variable
        train_data_rm = self.data.drop([self.protected_attribute, self.target_attribute], axis=1)
        train_data_rm_cols = list(train_data_rm.columns)
        cr = CorrelationRemover(sensitive_feature_ids=[self.protected_attribute], alpha=alpha)
        train_data_cr = cr.fit_transform(self.data.drop([self.target_attribute], axis=1))
        train_data_cr = pd.DataFrame(train_data_cr, columns=train_data_rm_cols)

        # complete data after correlation remover
        train_data_mitigated = pd.concat(
            [pd.DataFrame(self.data[self.target_attribute]), pd.DataFrame(self.data[self.protected_attribute]),
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

        # putting data in specific standardize form required by the package
        train_data_std = StandardDataset(self.data,
                                         label_name=self.target_attribute,
                                         favorable_classes=[self.favorable_label],
                                         protected_attribute_names=[self.protected_attribute],
                                         privileged_classes=[[self.privileged_class]])

        RW = Reweighing(unprivileged_groups=self.unprivileged_groups, privileged_groups=self.privileged_groups)
        # dataset_transf_train = RW.fit_transform(dataset_orig_train)
        RW = RW.fit(train_data_std)
        # train data after reweighing
        train_data_std_m = RW.transform(train_data_std)
        # train_data_mitigated = train_data_std_m.convert_to_dataframe()[0]
        # train_data_std_m.features - mitigated data features (i.e. without target values)
        # train_data_std_m.labels.ravel() - mitigated data target values
        # train_data_std_m.instance_weights - mitigated data weights for machine learning model

        # data after mitigation
        mitigated_dataset['weights'] = train_data_std_m.instance_weights
        # transform as an object
        mitigated_dataset['transform'] = RW

        return mitigated_dataset

    def _disp_removing_data(self, repair_level=0.8):
        """
        Transforming input using the 'DisparateImpactRemover' from aif360 pacakge.
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

        # putting data in specific standardize form required by the package
        train_data_std = BinaryLabelDataset(favorable_label=self.favorable_label,
                                            unfavorable_label=self.unfavorable_label,
                                            df=self.data,
                                            label_names=[self.target_attribute],
                                            protected_attribute_names=[self.protected_attribute])

        DIR = DisparateImpactRemover(repair_level=repair_level)
        train_data_std = DIR.fit_transform(train_data_std)
        train_data_mitigated = train_data_std.convert_to_dataframe()[0]

        # train data after mitigation
        mitigated_dataset['data'] = train_data_mitigated
        #  transform as an object
        mitigated_dataset['transform'] = DIR

        return mitigated_dataset
