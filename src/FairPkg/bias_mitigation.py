"""
Apply mitigation method
"""
import pandas as pd
import numpy as np
import logging

from dalex import Explainer
from dalex.fairness import resample
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class BiasMitigation:
    """

    """

    def __init__(self,
                 ml_model: object,
                 training_data: pd.DataFrame,
                 target_col: str,
                 favorable_label: int,
                 protected_col: str,
                 privileged_class: int,
                 unprivileged_class=None,
                 unfavorable_label=None):
        """
        Apply a mitigation method to data to make in more balanced.

        Args:
            ml_model (object):
                Target variable in the input data
            training_data (pd.DataFrame):
                Input data comprising 'training' and 'testing' data in dictionary format
            target_col (str):
                Target variable in the input data
            protected_col (str):
                Target variable in the input data
            privileged_class (int):
                Target variable in the input data
            unprivileged_class (int):
                Target variable in the input data
        """
        super(BiasMitigation, self).__init__()

        assert all(np.issubdtype(dtype, np.number) for dtype in training_data.dtypes)
        assert target_col in training_data.columns
        # assert isinstance(favorable_classes, (float, int))
        # assert isinstance(unfavorable_classes, (float, int))

        self.ml_model = ml_model
        self.training_data = training_data
        self.target_col = target_col
        self.favorable_label = favorable_label
        if unfavorable_label is None:
            _unfavorable_labels = list(set(self.training_data[target_col]))
            _unfavorable_labels.remove(self.favorable_label)
            self.unfavorable_label = _unfavorable_labels[0]  # just use one unfavorable label
            logger.debug(f"Computed unfavorable label {self.unfavorable_label}")
        else:
            self.unfavorable_label = unfavorable_label

        self.protected_col = protected_col
        self.privileged_class = privileged_class
        if unprivileged_class is None:
            _unprivileged_classes = list(set(self.training_data[protected_col]))
            _unprivileged_classes.remove(self.privileged_class)
            self.unprivileged_class = _unprivileged_classes[0]  # just use one unprivileged class
            logger.debug(f"Computed unprivileged class {self.unprivileged_class}")
        else:
            self.unprivileged_class = unprivileged_class

        self.unprivileged_groups = [{self.protected_col: [unprivileged_class]}]
        self.privileged_groups = [{self.protected_col: [privileged_class]}]

    def fit_transform(self, mitigation_method: str, cr_coeff=1, repair_level=0.8) -> dict:
        """
        Apply a mitigation method to data to make in more balanced.

        Args:
            mitigation_method (str):
                 Name of the mitigation method. Accepted values:
                 "resampling"
                 "resampling-preferential"
                 "reweighing"
                 "disparate-impact-remover"
                 "correlation-remover"
            cr_coeff (float):
                Correlation coefficient
            repair_level (float):
                Correlation coefficient
        Returns:
            dict:
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

        mitigated_dataset = {}
        if "resampling" in mitigation_method:
            mitigated_dataset = self.resampling_data(mitigation_method)
        elif mitigation_method == "correlation-remover":
            mitigated_dataset = self.cr_removing_data(cr_coeff)
        elif mitigation_method == "reweighing":
            mitigated_dataset = self.reweighing_data()
        elif mitigation_method == "disparate-impact-remover":
            mitigated_dataset = self.disp_removing_data(repair_level)

        mitigation_output = {mitigation_method: mitigated_dataset}
        return mitigation_output

    def resampling_data(self, mitigation_method: str) -> dict:
        """
        Resample the data using dalex module functions.

        Args:
            mitigation_method (str): Name of the mitigation method. Accepted values:
                "resampling",
                "resampling-uniform",
                "resampling-preferential"
        Returns:
            dict: Data after resampling and corresponding indexes
        """

        assert mitigation_method in ["resampling-uniform", "resampling", "resampling-preferential"], \
            "Incorrect resampling mitigation method. Valid methods are 'resampling-uniform', 'resampling', " \
            "'resampling-preferential'"

        mitigated_dataset = {}

        # Uniform resampling
        idx_resample = 0
        if (mitigation_method == "resampling-uniform") or (mitigation_method == "resampling"):
            idx_resample = resample(self.training_data[self.protected_col], self.training_data[self.target_col],
                                    type='uniform',
                                    verbose=False)
        # preferential resampling
        elif mitigation_method == "resampling-preferential":
            exp = Explainer(self.ml_model, self.training_data[self.training_data.columns.drop(self.target_col)].values,
                            self.training_data[self.target_col].values, verbose=False)
            idx_resample = resample(self.training_data[self.protected_col], self.training_data[self.target_col],
                                    type='preferential', verbose=False,
                                    probs=exp.y_hat)

        mitigated_data = self.training_data.iloc[idx_resample, :]
        # mitigated data
        mitigated_dataset['data'] = mitigated_data
        # resample index
        mitigated_dataset['index'] = idx_resample

        return mitigated_dataset

    def cr_removing_data(self, cr_coeff) -> dict:
        """
        correlation-remover (FairLearn)
        Args:
            cr_coeff (float):
                Correlation coefficient
        Returns:
            dict: Data after removing some samples and corresponding CorrelationRemover object
        """
        mitigated_dataset = {}

        # Getting correlation coefficient for mitigation_method 'correlation_remover'
        # The alpha parameter is used to control the level of filtering between the sensitive and non-sensitive features

        # remove the outcome variable and sensitive variable
        train_data_rm = self.training_data.drop([self.protected_col, self.target_col], axis=1)
        train_data_rm_cols = list(train_data_rm.columns)
        cr = CorrelationRemover(sensitive_feature_ids=[self.protected_col], alpha=cr_coeff)
        train_data_cr = cr.fit_transform(self.training_data.drop([self.target_col], axis=1))
        train_data_cr = pd.DataFrame(train_data_cr, columns=train_data_rm_cols)

        # complete data after correlation remover
        train_data_mitigated = pd.concat(
            [pd.DataFrame(self.training_data[self.target_col]), pd.DataFrame(self.training_data[self.protected_col]),
             train_data_cr], axis=1)

        mitigated_dataset['data'] = train_data_mitigated
        # correlation transform as an object
        mitigated_dataset['transform'] = cr

        return mitigated_dataset

    def reweighing_data(self) -> dict:
        """
        reweighing(AIF360)
        Returns:
            dict: Data after removing some samples and corresponding Reweighing object
        """

        mitigated_dataset = {}

        # putting data in specific standardize form required by the package
        train_data_std = StandardDataset(self.training_data,
                                         label_name=self.target_col,
                                         favorable_classes=[self.favorable_label],
                                         protected_attribute_names=[self.protected_col],
                                         privileged_classes=[self.privileged_class])

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

        import pdb
        pdb.set_trace()

        return mitigated_dataset

    def disp_removing_data(self, repair_level) -> dict:
        """
        disparate-impact-remover (AIF360)
        Args:
            repair_level (float):
                Correlation coefficient
        Returns:
            dict: Data after removing some samples and corresponding  DisparateImpactRemover object
        """

        mitigated_dataset = {}

        # putting data in specific standardize form required by the package
        train_data_std = BinaryLabelDataset(favorable_label=self.favorable_label,
                                            unfavorable_label=self.unfavorable_label,
                                            df=self.training_data,
                                            label_names=[self.target_col],
                                            protected_attribute_names=[self.protected_col])

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
    # Estimate prediction probability and predicted class of training data (Put empty dataframe for testing in order to estimate this)
    pred_class = mdl_clf.predict(X_test)
    pred_prob = mdl_clf.predict_proba(X_test)
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # Evaluate some scores
    auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    print(f"AUC = {auc}")

    accuracy = accuracy_score(y_test, pred_class)  # classification accuracy
    print(f"Accuracy = {accuracy}")

    # Create the BiasMitigation object to perform a bias mitigation
    bias_mitigation = BiasMitigation(ml_model=mdl_clf, training_data=training_data, target_col=target_var, favorable_label=1,
                                     protected_col='RACERETH', privileged_class=1)

    mitigation_method = "correlation-remover"
    # "resampling-preferential", "reweighing", "disparate-impact-remover", "correlation-remover"
    mitigation_res = bias_mitigation.fit_transform(mitigation_method=mitigation_method)

    mitigated_data = mitigation_res[mitigation_method]['data']
    X_train = mitigated_data.drop(columns=target_var)
    y_train = mitigated_data[target_var]

    # ReTrain Random Forest based on mitigated data
    mdl_clf = RandomForestClassifier(**param_ml)
    mdl_clf.fit(X_train, y_train)
    # Estimate prediction probability and predicted class of training data (Put empty dataframe for testing in order to estimate this)
    pred_class = mdl_clf.predict(X_test)
    pred_prob = mdl_clf.predict_proba(X_test)
    pred_prob = pred_prob[:, 1]  # keep probabilities for positive outcomes only

    # re-evaluate the scores
    new_auc = roc_auc_score(y_test, pred_prob)  # Area under a curve
    print(f"New AUC = {new_auc}")

    new_accuracy = accuracy_score(y_test, pred_class)  # classification accuracy
    print(f"New accuracy = {new_accuracy}")

