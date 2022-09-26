import logging
import pandas as pd
import dalex as dx
from dalex.fairness import resample
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover
from fairlearn.preprocessing import CorrelationRemover
# ===========================================================================================
# APPLY MITIGATION 
# ===========================================================================================
def mitigation_methods(data_input, target_var, mitigation_method, param_mitigation_method):
    """Apply mitigation"""
    # ---------------------------------------------------------------------------------------
    # 
    # INPUT
    # data_input (dictionary)       :   Input data comprising 'training' and 'testing' data in dictionary format                          
    #
    # target_var (string)           :   Target variable in the input data
    # 
    # mitigation_method (string)    :   Name of the mitigation method
    #                                       e.g. "resampling", "resampling-preferential", "reweighting", "disparate-impact-remover", "correlation-remover"
    # 
    # param_mitigation_method (dictionary)      :   Parameters for the mitigation method
    #                                                   param_mitigation_method  = {
    #                                                       'protected_var' : protected_var,
    #                                                       'privileged_classes': privileged_classes, 
    #                                                       'unprivileged_classes' : unprivileged_classes,
    #                                                       'favorable_classes' : favorable_classes,
    #                                                       'unfavorable_classes': unfavorable_classes,
    #                                                       'model': ml_output['model']        
    #                                                   }
    # 
    # OUTPUT
    # mitigation_output (dictionary of dictionary)    :   Data after mitigation and corresponding transforms/indexes
    # 
    # Note: 
    # 1. 'mitigation_output' key is mitigation method. Mitigated data and corresponding transform is stored as dictionary
    # 2. 'mitigation_output' method key shouldn't have both 'data' and 'model'    
    #       i.e. if method only changes data then key is 'data' and if method changes machine learning model than 'model' is in key                                  
    # ---------------------------------------------------------------------------------------
    model_ml=param_mitigation_method['model'] # trained machine learning model
    # pred_prob_data = param_mitigation_method['pred_prob_data']
    # pred_class_data = param_mitigation_method['pred_class_data']
    protected_var=param_mitigation_method['protected_var']
    privileged_classes=param_mitigation_method['privileged_classes']
    unprivileged_classes=param_mitigation_method['unprivileged_classes']
    favorable_classes=param_mitigation_method['favorable_classes']
    unfavorable_classes=param_mitigation_method['unfavorable_classes']
    # ---------------------------------------------------------------------------------------
    # DEFAULT VALUES
    # ---------------------------------------------------------------------------------------
    # mitigation_method: 'correlation_remover'
    if(not 'cr_coeff' in param_mitigation_method):
        cr_coeff=1 # correlation coefficient (alpha)
    else:
        cr_coeff=param_mitigation_method['cr_coeff'] 
        # The default value is 1, the alpha range is from 0 to 1
        # The alpha parameter is use to control the level of filtering between the sensitive and non-sensitive features
    # .......................................................................................
    # mitigation_method: 'disparate-impact-remover'
    if(not 'repair_level' in param_mitigation_method):
        repair_level=0.8
    else:
        repair_level=param_mitigation_method['repair_level']
    # ---------------------------------------------------------------------------------------
    # MITIGATION METHODS
    # 1. "resampling": "resampling-uniform", "resampling-preferential" (Dalex)
    # 2. "reweighting" (AIF360)
    # 3. "disparate-impact-remover" (AIF360)
    # 4. "correlation-remover" (FairLearn) 
    # ---------------------------------------------------------------------------------------
    mitigation_output = {}
    # ---------------------------------------------------------------------------------------
    if("resampling" in mitigation_method):
        mitigation_output[mitigation_method]={}
        # default type for resampling
        if((mitigation_method=="resampling-uniform") or (mitigation_method=="resampling")):
            # type1 - uniform
            idx_resample=resample(data_input[protected_var],data_input[target_var],type='uniform',verbose=False)
        elif(mitigation_method=="resampling-preferential"):
            # type2 - preferential
            exp = dx.Explainer(model_ml, data_input[data_input.columns.drop(target_var)].values, data_input[target_var].values, verbose=False)
            idx_resample = resample(data_input[protected_var], data_input[target_var], type = 'preferential', verbose = False, probs = exp.y_hat)
        data_input=data_input.iloc[idx_resample,:]
        # mitigated data
        mitigation_output[mitigation_method]['data'] = data_input
        # resample index
        mitigation_output[mitigation_method]['index'] = idx_resample
    # ---------------------------------------------------------------------------------------
    elif(mitigation_method ==  "correlation-remover"):
        mitigation_output[mitigation_method]={}
        # remove the outcome variable and sensitive variable
        data_input_rm = data_input.drop([protected_var, target_var], axis=1)
        data_input_rm_cols = list(data_input_rm.columns)
        cr = CorrelationRemover(sensitive_feature_ids=[protected_var], alpha=cr_coeff)
        data_input_cr = cr.fit_transform(data_input.drop([target_var], axis=1))
        data_input_cr = pd.DataFrame(data_input_cr,columns=data_input_rm_cols)
        # complete data after correlation remover
        data_input_mitigated = pd.concat([pd.DataFrame(data_input[target_var]), pd.DataFrame(data_input[protected_var]),data_input_cr], axis=1)        
        mitigation_output[mitigation_method]['data'] = data_input_mitigated
        # correlation transform as an object
        mitigation_output[mitigation_method]['transform'] = cr
    # ---------------------------------------------------------------------------------------  
    elif(mitigation_method ==  "reweighting"):
        mitigation_output[mitigation_method]={}
        # specific format e.g. [{'RACERETH': [2]}]
        unprivileged_groups = [{protected_var: unprivileged_classes}]
        privileged_groups = [{protected_var: privileged_classes}]
        # putting data in specific standardize form required by the package
        data_input_std = StandardDataset(data_input,
                            label_name=target_var,
                            favorable_classes=favorable_classes,
                            protected_attribute_names=[protected_var],
                            privileged_classes=[privileged_classes]) 
        RW = Reweighing(unprivileged_groups=unprivileged_groups,privileged_groups=privileged_groups)
        # dataset_transf_train = RW.fit_transform(dataset_orig_train)
        RW = RW.fit(data_input_std)
        # input data after reweighting
        data_input_std_m = RW.transform(data_input_std)
        # data_input_mitigated = data_input_std_m.convert_to_dataframe()[0]
        # data_input_std_m.features - mitigated data features (i.e. without target values)
        # data_input_std_m.labels.ravel() - mitigated data target values
        # data_input_std_m.instance_weights - mitigated data weights for machine learning model
        # input data after mitigation
        mitigation_output[mitigation_method]['model'] = data_input_std_m.instance_weights
        #  transforma as an object
        mitigation_output[mitigation_method]['transform'] = RW
        import pdb; pdb.set_trace()
    # --------------------------------------------------------------------------------------- 
    elif(mitigation_method ==  "disparate-impact-remover"):
        mitigation_output[mitigation_method]={}
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
        mitigation_output[mitigation_method]['data'] = data_input_mitigated
        #  transform as an object
        mitigation_output[mitigation_method]['transform'] = DIR
    # ---------------------------------------------------------------------------------------
    return mitigation_output        
# ===========================================================================================