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

# Add a custom logger for debugging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class BiasMitigation:
    """
    Bias mitigation class. Apply a mitigation method to data to make it more balanced.

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
            self.unprivileged_class = _unprivileged_classes[0]  # just use one unprivileged class
            logger.debug(f"Computed unprivileged class {self.unprivileged_class}")
        else:
            self.unprivileged_class = unprivileged_class

        self.unprivileged_groups = [{self.protected_attribute: [self.unprivileged_class]}]
        self.privileged_groups = [{self.protected_attribute: [self.privileged_class]}]

    def fit_transform(self, mitigation_method, cr_coeff=1, repair_level=0.8):
        """
        Apply a mitigation method to data to make in more balanced.

        Parameters
        ----------
        mitigation_method : str
             Name of the mitigation method. Accepted values:
             "resampling"
             "resampling-preferential"
             "reweighing"
             "disparate-impact-remover"
             "correlation-remover"
        cr_coeff : float, default=1
            Correlation coefficient
        repair_level : float, default=0.8
            Correlation coefficient
        Returns
        ----------
        T : dictionary-like of shape
            Data after mitigation and corresponding transforms/indexes
            Notes:
            'mitigation_output' key is mitigation method. Mitigated data and corresponding transform is stored as
            dictionary
            'mitigation_output' method key shouldn't have both 'data' and 'model' i.e. if method only changes data then
             key is 'data' and if method changes machine learning model than 'model' is in key

        """

        assert mitigation_method in ["resampling-uniform", "resampling", "resampling-preferential",
                                     "correlation-remover",
                                     "reweighing", "disparate-impact-remover"], "Incorrect mitigation method."
        # TODO check cr_coeff and repair_level values

        mitigated_dataset = {}
        if "resampling" in mitigation_method:
            mitigated_dataset = self._resampling_data(mitigation_method)
        elif mitigation_method == "correlation-remover":
            mitigated_dataset = self._cr_removing_data(cr_coeff)
        elif mitigation_method == "reweighing":
            mitigated_dataset = self._reweighing_model()
        elif mitigation_method == "disparate-impact-remover":
            mitigated_dataset = self._disp_removing_data(repair_level)

        return mitigated_dataset

    def _resampling_data(self, mitigation_method):
        """
        Resample the data using dalex module functions.

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
            Data after resampling and corresponding indexes
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

    def _cr_removing_data(self, cr_coeff=1):
        """
        correlation-remover (FairLearn)
        Parameters
        ----------
        cr_coeff : float, default=1
            Correlation coefficient
        Returns
        ----------
        T : dictionary-like of shape
            Data after removing some samples and corresponding CorrelationRemover object
        """
        mitigated_dataset = {}

        # Getting correlation coefficient for mitigation_method 'correlation_remover'
        # The alpha parameter is used to control the level of filtering between the sensitive and non-sensitive features

        # remove the outcome variable and sensitive variable
        train_data_rm = self.data.drop([self.protected_attribute, self.target_attribute], axis=1)
        train_data_rm_cols = list(train_data_rm.columns)
        cr = CorrelationRemover(sensitive_feature_ids=[self.protected_attribute], alpha=cr_coeff)
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
        reweighing(AIF360)
        Returns
        ----------
        T : dictionary-like of shape
            Data after removing some samples and corresponding Reweighing object
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
        mitigated_dataset['model'] = train_data_std_m.instance_weights
        # transform as an object
        mitigated_dataset['transform'] = RW

        #mport pdb
        #pdb.set_trace()

        return mitigated_dataset

    def _disp_removing_data(self, repair_level=0.8):
        """
        disparate-impact-remover (AIF360)
        Parameters
        ----------
        repair_level : float, default=0.8
            Correlation coefficient
        Returns
        ----------
        T : dictionary-like of shape
            Data after removing some samples and corresponding  DisparateImpactRemover object
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
    # Estimate prediction probability and predicted class of training data (Put empty dataframe for testing in order to
    # estimate this)
    pred_class = mdl_clf.predict(X_test)
    pred_prob = mdl_clf.predict_proba(X_test)
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # Evaluate some scores
    auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    print(f"AUC = {auc}")

    accuracy = accuracy_score(y_test, pred_class)  # classification accuracy
    print(f"Accuracy = {accuracy}")

    # Create the BiasMitigation object to perform a bias mitigation
    bias_mitigation = BiasMitigation(ml_model=mdl_clf, data=training_data, target_attribute=target_var,
                                     protected_attribute='RACERETH', privileged_class=1)

    mitigation_method = "disparate-impact-remover"
    # "resampling-uniform", "resampling", "resampling-preferential", "correlation-remover", "reweighing",
    # "disparate-impact-remover"

    # For mitigation_method = reweighing, the result is model weights on data. How to solve that???
    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)

    if mitigation_method != "reweighing":
        mitigated_data = mitigation_res['data']
        X_train = mitigated_data.drop(columns=target_var)
        y_train = mitigated_data[target_var]

        # ReTrain Random Forest based on mitigated data
        mdl_clf = RandomForestClassifier(**param_ml)
        mdl_clf.fit(X_train, y_train)
    else:
        mitigated_weights = mitigation_res['model']
        mdl_clf.fit(X_train, y_train, sample_weight=mitigated_weights)
    # Estimate prediction probability and predicted class of training data (Put empty dataframe for testing in order to
    # estimate this)
    pred_class = mdl_clf.predict(X_test)
    pred_prob = mdl_clf.predict_proba(X_test)
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # re-evaluate the scores
    new_auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    print(f"New AUC = {new_auc}")

    new_accuracy = accuracy_score(y_test, pred_class)  # classification accuracy
    print(f"New accuracy = {new_accuracy}")
