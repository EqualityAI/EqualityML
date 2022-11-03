"""
Apply mitigation method
"""
import pandas as pd
import dalex as dx
from dalex.fairness import resample
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover
import logging

logger = logging.getLogger(__name__)


def mitigation_methods(
        data_input: pd.DataFrame,
        target_var: str,
        mitigation_method: str,
        param_mitigation_method: dict) -> dict:
    """
    Apply a mitigation method to data to make in more balanced.

    Args:
        data_input (pd.DataFrame):
            Input data comprising 'training' and 'testing' data in dictionary format
        target_var (str):
            Target variable in the input data
        mitigation_method (str):
             Name of the mitigation method. Accepted values:
             "resampling"
             "resampling-preferential"
             "reweighting"
             "disparate-impact-remover"
             "correlation-remover"
        param_mitigation_method (dict):
            Parameters for the mitigation method.
            param_mitigation_method  = {
                                        'protected_var' : protected_var,
                                        'privileged_classes': privileged_classes,
                                        'unprivileged_classes' : unprivileged_classes,
                                        'favorable_classes' : favorable_classes,
                                        'unfavorable_classes': unfavorable_classes,
                                        'model': ml_output['model']
                                       }
    Returns:
        dict:
            Data after mitigation and corresponding transforms/indexes
            Notes:
            'mitigation_output' key is mitigation method. Mitigated data and corresponding transform is stored as
            dictionary
            'mitigation_output' method key shouldn't have both 'data' and 'model' i.e. if method only changes data then
             key is 'data' and if method changes machine learning model than 'model' is in key

    """

    # Extracting parameters for the mitigation method
    # pred_prob_data = param_mitigation_method['pred_prob_data']
    # pred_class_data = param_mitigation_method['pred_class_data']

    assert mitigation_method in ["resampling-uniform", "resampling", "resampling-preferential", "correlation-remover",
                                 "reweighting", "disparate-impact-remover"], "Incorrect mitigation method."

    mitigated_dataset = {}
    if "resampling" in mitigation_method:
        mitigated_dataset = resampling_data(data_input, target_var, mitigation_method, param_mitigation_method)
    elif mitigation_method == "correlation-remover":
        mitigated_dataset = cr_removing_data(data_input, target_var, param_mitigation_method)
    elif mitigation_method == "reweighting":
        mitigated_dataset = reweighting_data(data_input, target_var, param_mitigation_method)
    elif mitigation_method == "disparate-impact-remover":
        mitigated_dataset = disp_removing_data(data_input, target_var, param_mitigation_method)

    mitigation_output = {mitigation_method: mitigated_dataset}
    return mitigation_output


def resampling_data(
        data_input: pd.DataFrame,
        target_var: str,
        mitigation_method: str,
        param_mitigation_method: dict) -> dict:
    """
    Resample the data using dalex module functions.

    Args:
        data_input (pd.DataFrame): Input data comprising 'training' and 'testing' data in dictionary format
        target_var (str): Target variable in the input data
        mitigation_method (str): Name of the mitigation method. Accepted values:
            "resampling",
            "resampling-uniform",
            "resampling-preferential"
        param_mitigation_method (dict): Parameters for the mitigation method.
    Returns:
        dict: Data after resampling and corresponding indexes
    """

    assert mitigation_method in ["resampling-uniform", "resampling", "resampling-preferential"], \
        "Incorrect resampling mitigation method. Valid methods are 'resampling-uniform', 'resampling', " \
        "'resampling-preferential'"

    mitigated_dataset = {}
    # Extracting parameters for the mitigation method
    protected_var = param_mitigation_method['protected_var']
    # trained machine learning model
    model_ml = param_mitigation_method['model']

    # Uniform resampling
    idx_resample = 0
    if (mitigation_method == "resampling-uniform") or (mitigation_method == "resampling"):
        idx_resample = resample(data_input[protected_var], data_input[target_var], type='uniform', verbose=False)
    # preferential resampling
    elif mitigation_method == "resampling-preferential":
        exp = dx.Explainer(model_ml, data_input[data_input.columns.drop(target_var)].values,
                           data_input[target_var].values, verbose=False)
        idx_resample = resample(data_input[protected_var], data_input[target_var], type='preferential', verbose=False,
                                probs=exp.y_hat)

    data_input = data_input.iloc[idx_resample, :]
    # mitigated data
    mitigated_dataset['data'] = data_input
    # resample index
    mitigated_dataset['index'] = idx_resample

    return mitigated_dataset


def cr_removing_data(data_input: pd.DataFrame, target_var: str, param_mitigation_method: dict) -> dict:
    """
    correlation-remover (FairLearn)

    Args:
        data_input (pd.DataFrame): Input data comprising 'training' and 'testing' data in dictionary format
        target_var (str): Target variable in the input data
        param_mitigation_method (dict): Parameters for the mitigation method.
    Returns:
        dict: Data after removing some samples and corresponding CorrelationRemover object
    """
    mitigated_dataset = {}
    # Extracting parameters for the mitigation method
    protected_var = param_mitigation_method['protected_var']

    # Getting correlation coefficient for mitigation_method 'correlation_remover'
    if 'cr_coeff' not in param_mitigation_method:
        # correlation coefficient (alpha)
        cr_coeff = 1
    else:
        # The default value is 1, the alpha range is from 0 to 1
        # The alpha parameter is used to control the level of filtering between the sensitive and non-sensitive features
        cr_coeff = param_mitigation_method['cr_coeff']

    # remove the outcome variable and sensitive variable
    data_input_rm = data_input.drop([protected_var, target_var], axis=1)
    data_input_rm_cols = list(data_input_rm.columns)
    cr = CorrelationRemover(sensitive_feature_ids=[protected_var], alpha=cr_coeff)
    data_input_cr = cr.fit_transform(data_input.drop([target_var], axis=1))
    data_input_cr = pd.DataFrame(data_input_cr, columns=data_input_rm_cols)

    # complete data after correlation remover
    data_input_mitigated = pd.concat(
        [pd.DataFrame(data_input[target_var]), pd.DataFrame(data_input[protected_var]), data_input_cr], axis=1)

    mitigated_dataset['data'] = data_input_mitigated
    # correlation transform as an object
    mitigated_dataset['transform'] = cr

    return mitigated_dataset


def reweighting_data(data_input: pd.DataFrame, target_var: str, param_mitigation_method: dict) -> dict:
    """
    reweighting(AIF360)
    Args:
        data_input (pd.DataFrame): Input data comprising 'training' and 'testing' data in dictionary format
        target_var (str): Target variable in the input data
        param_mitigation_method (dict): Parameters for the mitigation method.
    Returns:
        dict: Data after removing some samples and corresponding Reweighing object
    """

    mitigated_dataset = {}
    # Extracting parameters for the mitigation method
    protected_var = param_mitigation_method['protected_var']
    privileged_classes = param_mitigation_method['privileged_classes']
    unprivileged_classes = param_mitigation_method['unprivileged_classes']
    favorable_classes = param_mitigation_method['favorable_classes']

    # specific format e.g. [{'RACERETH': [2]}]
    unprivileged_groups = [{protected_var: unprivileged_classes}]
    privileged_groups = [{protected_var: privileged_classes}]

    # putting data in specific standardize form required by the package
    data_input_std = StandardDataset(data_input,
                                     label_name=target_var,
                                     favorable_classes=favorable_classes,
                                     protected_attribute_names=[protected_var],
                                     privileged_classes=[privileged_classes])

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    # dataset_transf_train = RW.fit_transform(dataset_orig_train)
    RW = RW.fit(data_input_std)
    # input data after reweighting
    data_input_std_m = RW.transform(data_input_std)
    # data_input_mitigated = data_input_std_m.convert_to_dataframe()[0]
    # data_input_std_m.features - mitigated data features (i.e. without target values)
    # data_input_std_m.labels.ravel() - mitigated data target values
    # data_input_std_m.instance_weights - mitigated data weights for machine learning model

    # data after mitigation
    mitigated_dataset['model'] = data_input_std_m.instance_weights
    # transform as an object
    mitigated_dataset['transform'] = RW

    import pdb
    pdb.set_trace()

    return mitigated_dataset


def disp_removing_data(data_input: pd.DataFrame, target_var: str, param_mitigation_method: dict) -> dict:
    """
    disparate-impact-remover (AIF360)
    Args:
        data_input (pd.DataFrame): Input data comprising 'training' and 'testing' data in dictionary format
        target_var (str): Target variable in the input data
        param_mitigation_method (dict): Parameters for the mitigation method.
    Returns:
        dict: Data after removing some samples and corresponding  DisparateImpactRemover object
    """

    mitigated_dataset = {}
    # Extracting parameters for the mitigation method
    protected_var = param_mitigation_method['protected_var']
    favorable_classes = param_mitigation_method['favorable_classes']
    unfavorable_classes = param_mitigation_method['unfavorable_classes']

    if 'repair_level' not in param_mitigation_method:
        repair_level = 0.8
    else:
        repair_level = param_mitigation_method['repair_level']

    # putting data in specific standardize form required by the package
    data_input_std = BinaryLabelDataset(favorable_label=favorable_classes[0],
                                        unfavorable_label=unfavorable_classes[0],
                                        df=data_input,
                                        label_names=[target_var],
                                        protected_attribute_names=[protected_var])

    DIR = DisparateImpactRemover(repair_level=repair_level)
    data_input_std = DIR.fit_transform(data_input_std)
    data_input_mitigated = data_input_std.convert_to_dataframe()[0]

    # input data after mitigation
    mitigated_dataset['data'] = data_input_mitigated
    #  transform as an object
    mitigated_dataset['transform'] = DIR

    return mitigated_dataset


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    # First train a Machine Learning model with the training data

    # Read training and testing data.
    target_var = "HOS"
    training_data = pd.read_csv("fairness_data/data_train.csv")
    X_train = training_data.drop(columns=target_var)
    y_train = training_data[target_var]
    testing_data = pd.read_csv("fairness_data/data_test.csv")
    X_test = testing_data.drop(columns=target_var)

    # Train a Random Forest model
    param_ml = {
        "n_estimators": 500,  # Number of trees in the forest
        "min_samples_split": 6,  # Minimum number of samples required  to split an internal node
        "random_state": 0
    }
    mdl_clf = RandomForestClassifier(**param_ml)
    mdl_clf.fit(X_train, y_train)

    # Choose a mitigation-method and its parameters
    mitigation_method = "correlation-remover"
    # "resampling-preferential", "reweighting", "disparate-impact-remover", "correlation-remover"
    param_mitigation_method = {
        'protected_var': 'RACERETH',
        'privileged_classes': [1],
        'unprivileged_classes': [2],
        'favorable_classes': [1],
        'unfavorable_classes': [0],
        'model': mdl_clf
    }
    mitigation_res = mitigation_methods(training_data, target_var, mitigation_method, param_mitigation_method)
