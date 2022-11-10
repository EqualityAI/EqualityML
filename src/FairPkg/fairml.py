import pandas as pd
import numpy as np
import logging

from dalex import Explainer
from dalex.fairness import resample
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.metrics import ClassificationMetric
from fairlearn.preprocessing import CorrelationRemover

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class FairML:
    """

    """

    def __init__(self,
                 ml_model: object,
                 train_data: pd.DataFrame,
                 target_var: str,
                 protected_var: str,
                 privileged_classes: int,
                 unprivileged_classes: int,
                 favorable_classes: int,
                 unfavorable_classes: int,
                 pred_prob_data: int,
                 pred_class_data: int):
        """
        Apply a mitigation method to data to make in more balanced.

        Args:
            ml_model (object):
                Target variable in the input data
            train_data (pd.DataFrame):
                Input data comprising 'training' and 'testing' data in dictionary format
            target_var (str):
                Target variable in the input data
            protected_var (str):
                Target variable in the input data
            privileged_classes (int):
                Target variable in the input data
            unprivileged_classes (int):
                Target variable in the input data
            pred_prob_data (int):
                Target variable in the input data
            pred_class_data (int):
                Target variable in the input data
        """
        super(FairML, self).__init__()

        assert all(np.issubdtype(dtype, np.number) for dtype in train_data.dtypes)
        assert target_var in train_data.columns
        # assert isinstance(favorable_classes, (float, int))
        # assert isinstance(unfavorable_classes, (float, int))

        self.ml_model = ml_model
        self.train_data = train_data
        self.target_var = target_var
        self.protected_var = protected_var
        self.favorable_classes = favorable_classes
        self.unfavorable_classes = unfavorable_classes
        self.privileged_classes = privileged_classes
        self.pred_prob_data = pred_prob_data
        self.pred_class_data = pred_class_data

        # specific format e.g. [{'RACERETH': [2]}]
        self.unprivileged_groups = [{self.protected_var: [unprivileged_classes]}]
        self.privileged_groups = [{self.protected_var: [privileged_classes]}]

    def mitigation_methods(self, mitigation_method: str, cr_coeff=1, repair_level=0.8) -> dict:
        """
        Apply a mitigation method to data to make in more balanced.

        Args:
            mitigation_method (str):
                 Name of the mitigation method. Accepted values:
                 "resampling"
                 "resampling-preferential"
                 "reweighting"
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
                                     "reweighting", "disparate-impact-remover"], "Incorrect mitigation method."

        mitigated_dataset = {}
        if "resampling" in mitigation_method:
            mitigated_dataset = self.resampling_data(mitigation_method)
        elif mitigation_method == "correlation-remover":
            mitigated_dataset = self.cr_removing_data(cr_coeff)
        elif mitigation_method == "reweighting":
            mitigated_dataset = self.reweighting_data()
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
            idx_resample = resample(self.train_data[self.protected_var], self.train_data[self.target_var],
                                    type='uniform',
                                    verbose=False)
        # preferential resampling
        elif mitigation_method == "resampling-preferential":
            exp = Explainer(self.ml_model, self.train_data[self.train_data.columns.drop(self.target_var)].values,
                            self.train_data[self.target_var].values, verbose=False)
            idx_resample = resample(self.train_data[self.protected_var], self.train_data[self.target_var],
                                    type='preferential', verbose=False,
                                    probs=exp.y_hat)

        data_input = self.train_data.iloc[idx_resample, :]
        # mitigated data
        mitigated_dataset['data'] = data_input
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
        train_data_rm = self.train_data.drop([self.protected_var, self.target_var], axis=1)
        train_data_rm_cols = list(train_data_rm.columns)
        cr = CorrelationRemover(sensitive_feature_ids=[self.protected_var], alpha=cr_coeff)
        train_data_cr = cr.fit_transform(self.train_data.drop([self.target_var], axis=1))
        train_data_cr = pd.DataFrame(train_data_cr, columns=train_data_rm_cols)

        # complete data after correlation remover
        train_data_mitigated = pd.concat(
            [pd.DataFrame(self.train_data[self.target_var]), pd.DataFrame(self.train_data[self.protected_var]),
             train_data_cr], axis=1)

        mitigated_dataset['data'] = train_data_mitigated
        # correlation transform as an object
        mitigated_dataset['transform'] = cr

        return mitigated_dataset

    def reweighting_data(self) -> dict:
        """
        reweighting(AIF360)
        Returns:
            dict: Data after removing some samples and corresponding Reweighing object
        """

        mitigated_dataset = {}

        # putting data in specific standardize form required by the package
        train_data_std = StandardDataset(self.train_data,
                                         label_name=self.target_var,
                                         favorable_classes=[self.favorable_classes],
                                         protected_attribute_names=[self.protected_var],
                                         privileged_classes=[self.privileged_classes])

        RW = Reweighing(unprivileged_groups=self.unprivileged_groups, privileged_groups=self.privileged_groups)
        # dataset_transf_train = RW.fit_transform(dataset_orig_train)
        RW = RW.fit(train_data_std)
        # train data after reweighting
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
        train_data_std = BinaryLabelDataset(favorable_label=self.favorable_classes,
                                            unfavorable_label=self.unfavorable_classes,
                                            df=self.train_data,
                                            label_names=[self.target_var],
                                            protected_attribute_names=[self.protected_var])

        DIR = DisparateImpactRemover(repair_level=repair_level)
        train_data_std = DIR.fit_transform(train_data_std)
        train_data_mitigated = train_data_std.convert_to_dataframe()[0]

        # train data after mitigation
        mitigated_dataset['data'] = train_data_mitigated
        #  transform as an object
        mitigated_dataset['transform'] = DIR

        return mitigated_dataset

    def fairness_metrics(self, data: pd.DataFrame, metric_name='all', cutoff=0.5) -> dict:
        """
        Fairness metric evaluation using protected variable, privileged class, etc.

        Args:
            data (pd.DataFrame):
                data in DataFrame format
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
            cutoff (float):
                Threshold value used for the machine learning classifier
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
            data_std = BinaryLabelDataset(favorable_label=self.favorable_classes,
                                          unfavorable_label=self.unfavorable_classes,
                                          df=data,
                                          label_names=[self.target_var],
                                          protected_attribute_names=[self.protected_var],
                                          unprivileged_protected_attributes=self.unprivileged_groups)

            # fairness metric computation
            data_pred = data_std.copy()
            data_pred.scores = self.pred_prob_data  # predicted  probability
            data_pred.labels = self.pred_class_data  # predicted class
            cm_pred_data = ClassificationMetric(data_std,
                                                data_pred,
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
            X_data = data.drop(columns=self.target_var)
            # target variable
            y_data = data[[target_var]]

            # create an explainer
            exp = Explainer(self.ml_model, X_data, y_data, verbose=False)
            # define protected variable and privileged group
            protected_vec = X_data[self.protected_var]

            logger.critical('Machine learning model threshold: {}'.format(cutoff))

            fairness_object = exp.model_fairness(protected=protected_vec, privileged=str(self.privileged_classes),
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

    # Create the FairML object to evaluate the fairness metric and perform a bias mitigation
    param_mitigation_method = {
        'train_data': training_data,
        'target_var': target_var,
        'pred_prob_data': pred_prob,
        'pred_class_data': pred_class,
        'protected_var': 'RACERETH',
        'privileged_classes': 1,
        'unprivileged_classes': 2,
        'favorable_classes': 1,
        'unfavorable_classes': 0,
        'ml_model': mdl_clf
    }
    fair_ml = FairML(**param_mitigation_method)

    # Fairness metric
    fairness_metric_score = fair_ml.fairness_metrics(testing_data)
    print(fairness_metric_score)

    # Bias mitigation
    mitigation_method = "correlation-remover"
    # "resampling-preferential", "reweighting", "disparate-impact-remover", "correlation-remover"
    mitigation_res = fair_ml.mitigation_methods(mitigation_method)

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

    # Re-evaluate the fairness metric
    fair_ml.pred_prob_data = pred_prob
    fair_ml.pred_class_data = pred_class
    fair_ml.ml_model = mdl_clf
    new_fairness_metric_score = fair_ml.fairness_metrics(testing_data)
    print(new_fairness_metric_score)
