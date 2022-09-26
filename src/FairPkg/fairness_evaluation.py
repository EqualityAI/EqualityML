from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import dalex as dx
import logging
# ===========================================================================================
#                         FAIRNESS EVALUATION METRICS
# ===========================================================================================
def fairness_metrics(data_input, target_var, metric_name, param_fairness_metric):
    """Fairness metric evaluation using protected variable, privileged class, etc. """
    # 
    # INPUT
    # data_input (DataFrame             : input data (e.g. testing data) for the evaluation of fairness metric
    # target_var  (string)              : target variable
    # metric_name (string)              : fairness metric name   
    #                                            Fairness Metrics: 
    #                                               1. 'treatment_equality_ratio':
    #                                               2. 'treatment_equality_diff': 
    #                                               3. 'balance_positive_class': Balance for positive class
    #                                               4. 'balance_negative_class': Balance for negative class
    #                                               5. 'equal_opportunity_ratio': Equal opportunity ratio
    #                                               6. 'accuracy_equality_ratio': Accuracy equality ratio
    #                                               7. 'predictive_parity_ratio':  Predictive parity ratio
    #                                               8. 'predictive_equality_ratio': Predictive equality ratio
    #                                               9. 'statistical_parity_ratio': Statistical parity ratio 
    # 
    # param_fairness_metric (dictionary): parameters that are required for the evaluation of fairness metric
    #                                           e.g. param_fairness_metric = {
                                                        # 'pred_prob_data' : ml_output['probability'],
                                                        # 'pred_class_data': ml_output['class'],
                                                        # 'protected_var': protected_var, 
                                                        # 'privileged_classes': privileged_classes, 
                                                        # 'unprivileged_classes' : unprivileged_classes,
                                                        # 'favorable_classes' : favorable_classes,
                                                        # 'unfavorable_classes': unfavorable_classes,
                                                        # 'model':  ml_output['model'] 
                                                        # }
    # 
    # OUTPUT
    # fairness_metric_score (dictionary): fairness metric value
    # ---------------------------------------------------------------------------- 
    model_ml = param_fairness_metric['model'] # trained machine learning model object
    pred_prob_data = param_fairness_metric['pred_prob_data'] # predicted probability of the input data
    pred_class_data = param_fairness_metric['pred_class_data'] # predicted class of the input data
    protected_var = param_fairness_metric['protected_var'] # protected variable
    privileged_classes = param_fairness_metric['privileged_classes'][0] # privileged classes of the protected variable
    unprivileged_classes = param_fairness_metric['unprivileged_classes'][0] # unprivileged classes of the protected variable
    favorable_classes = param_fairness_metric['favorable_classes'][0]
    unfavorable_classes = param_fairness_metric['unfavorable_classes'][0]
    # ---------------------------------------------------------------------------- 
    #  PACKAGE: AIF360  
    method1_list = ['treatment_equality_ratio', 'treatment_equality_diff', 'balance_positive_class',  'balance_negative_class']
    #  PACKAGE: DALEX
    method2_list = ['equal_opportunity_ratio', 'accuracy_equality_ratio', 'predictive_parity_ratio','predictive_equality_ratio', 'statistical_parity_ratio']
    fairness_metric_score = {}
    # ---------------------------------------------------------------------------- 
    if((metric_name in method1_list) or (metric_name.lower() == 'all')):
        # define privileged and unprivileged groups
        unprivileged_groups = [{protected_var: unprivileged_classes}]
        privileged_groups = [{protected_var: privileged_classes}]
        # create a dataset according to structure required by the package AIF360
        data_input_std = BinaryLabelDataset(favorable_label=favorable_classes,
                                        unfavorable_label=unfavorable_classes,
                                        df=data_input,
                                        label_names=[target_var],
                                        protected_attribute_names=[protected_var],
                                        unprivileged_protected_attributes=unprivileged_groups)
        # fairness metric computation
        data_input_pred = data_input_std.copy()
        data_input_pred.scores = pred_prob_data # predicted  probability
        data_input_pred.labels = pred_class_data # predicted class
        cm_pred_data = ClassificationMetric(data_input_std,
                                        data_input_pred,
                                        unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        # Treatment equality
        # Note that both treatment equality ratio and treatment equality difference are calculated
        prev_rat = cm_pred_data.num_false_negatives(True) / cm_pred_data.num_false_positives(True)
        unprev_rat = cm_pred_data.num_false_negatives(False) / cm_pred_data.num_false_positives(False)
        treatment_equality_ratio = unprev_rat / prev_rat
        treatment_equality_diff = unprev_rat - prev_rat
        if((metric_name.lower() == 'treatment_equality_ratio') or (metric_name.lower() == 'all')):
            fairness_metric_score['treatment_equality_ratio'] = treatment_equality_ratio
        if((metric_name.lower() == 'treatment_equality_difference') or (metric_name.lower() == 'all')):
            fairness_metric_score['treatment_equality_difference'] = treatment_equality_diff
        if((metric_name.lower() == 'balance_negative_class') or (metric_name.lower() == 'all')):
            fairness_metric_score['balance_negative_class'] = cm_pred_data.ratio(cm_pred_data.generalized_false_positive_rate)
        if((metric_name.lower() == 'balance_positive_class') or (metric_name.lower() == 'all')):
            fairness_metric_score['balance_positive_class'] = cm_pred_data.ratio(cm_pred_data.generalized_true_positive_rate)
    # -------------------------------------------------------------------------- 
    if((metric_name in method2_list) | (metric_name.lower() == 'all')):
        X_data = data_input.drop(columns = target_var) # features
        y_data = data_input[[target_var]] # target variable
        # create an explainer
        exp = dx.Explainer(model_ml, X_data, y_data,verbose=False)
        # define protected variable and privileged group
        protected_vec = X_data[protected_var]
        # fairness evaluation
        if('cutoff' not in  param_fairness_metric):
            cutoff=0.5 # Threshold value used for the machine learning classifier
        else:
            cutoff=param_fairness_metric['cutoff']
        logging.critical('Machine learning model threshold: {}'.format(cutoff))
        fobject = exp.model_fairness(protected=protected_vec, privileged=privileged_classes, cutoff=cutoff)
        fairness_result = fobject.result
        if((metric_name.lower() == 'equal_opportunity_ratio') or (metric_name.lower() == 'all')):
            #NOTE: check index location for different values of privileged class
            #TEST: Does location is dependent on the value of the privileged class?
            fairness_metric_score['equal_opportunity_ratio'] = fairness_result['TPR'][1]
        if((metric_name.lower() == 'accuracy_equality_ratio') or (metric_name.lower() == 'all')):
            fairness_metric_score['accuracy_equality_ratio'] = fairness_result['ACC'][1]
        if((metric_name.lower() == 'predictive_parity_ratio') or (metric_name.lower() == 'all')):
            fairness_metric_score['predictive_parity_ratio'] = fairness_result['PPV'][1]
        if((metric_name.lower() == 'predictive_equality_ratio') or (metric_name.lower() == 'all')):
            fairness_metric_score['predictive_equality_ratio'] = fairness_result['FPR'][1]
        if((metric_name.lower() == 'statistical_parity_ratio') or (metric_name.lower() == 'all')):
            fairness_metric_score['statistical_parity_ratio'] = fairness_result['STP'][1]
    # -------------------------------------------------------------------------- 
    return fairness_metric_score   
#===============================================================================
            

    