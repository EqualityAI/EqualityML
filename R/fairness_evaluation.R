#===============================================================================
#                         FAIRNESS METRICS
#===============================================================================
fairness_metric <- function(ml_model_trained, data_input, target_var, param_fairness_metric){
  # Calculate fairness metric based on protected variable, privileged classs, etc. 
  # 
  # INPUT
  # ml_model_trained (list)   : Trained machine learning model
  # data_input (list)         : Data for calculating fairness metric
  # target_var (character)    : Target variable
  # param_fairness_metric     : Parameters for calculating fairness metrics
  #                               1. Protected variable : "protected"  
  #                               2. Privileged class from protected variable : "privileged"
  #
  # OUTPUT
  # fairness_score            : Fairness metric score
  #
  # EXAMPLE
  # param_fairness_metric = list("protected" = protected_var, "privileged" = privileged_class)
  # fairness_score <- fairness_metric(ml_output$model, data_clean$testing, target_var, param_fairness_metric)
  # ----------------------------------------------------------------------------
  protected_var <- param_fairness_metric$protected
  privileged_class <- param_fairness_metric$privileged
  
  # conversion of targeted variable to numeric
  data_input[[target_var]] <- as.numeric(as.character(data_input[[target_var]]))
  
  # Check: ignore protected variable for model explanation
  if("ignore_protected"  %in% names(param_fairness_metric)){
    if(param_fairness_metric$ignore_protected == TRUE){
      # removing protected variable from model explanation calculation
      data_explain <- var_rem(data_input, param_fairness_metric$protected)
      print('Warning: Protected variable not included in model explanation')
    }
    else{
      data_explain <- data_input
    }
    mdl_explain <- model_explain(ml_model_trained, data_explain, target_var)
  }
  fairness_score <- model_fairness(mdl_explain$explain, data_input, protected_var, privileged_class)
  
  predictive_equality <- fairness_score$object$fairness_check_data$score[3]
  equal_opportunity <- fairness_score$object$fairness_check_data$score[4]
  statistical_parity <- fairness_score$object$fairness_check_data$score[5]
  
  fairness_score <- list("Predictive Equality" = predictive_equality, 
                         "Equal Oopportunity" = equal_opportunity,
                         "Statistical Parity" = statistical_parity)
  
  return(fairness_score)
}
#===============================================================================
# MODEL EXPLAINER
#===============================================================================
model_explain <- function(ml_model_trained, data_input, target_var){
  # model_explainer <- DALEX::explain(ml_model_trained, data = training_data[, -1], y = training_data$readmitted) 
  model_explainer <- DALEX::explain(ml_model_trained, data = data_input[, -1], 
                               y = get(target_var[1], data_input), colorize = FALSE) 
  results = list("explain" = model_explainer)
  return(results)
}
#===============================================================================
# FAIRNESS CHECK
#===============================================================================
model_fairness <- function(mdl_explain, training_data, protected_var, privileged_class){
  fobject <- fairness_check(mdl_explain,
                      protected = get(protected_var[1], training_data),
                      privileged = privileged_class)
  results = list("object" = fobject)
  return(results)
}
  