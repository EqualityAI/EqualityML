#===============================================================================
#                         FAIRNESS METRICS
#===============================================================================
fairness_metric <- function(ml_model, input_data, target_variable, protected_variable, privileged_class, ignore_protected = TRUE){
  # Calculate fairness metric based on protected variable, privileged classs, etc. 
  # 
  # INPUT
  # ml_model (list)                 : Trained machine learning model
  # input_data (list)               : Data for calculating fairness metric
  # target_variable (character)     : Target variable
  # protected_variable (character)  : Protected variable
  # privileged_class (character)    : Privileged class from protected variable : "privileged"
  # OUTPUT
  # fairness_score            : Fairness metric score
  #
  # EXAMPLE
  # fairness_score <- fairness_metric(ml_output$model, data_clean$testing, target_variable, protected_variable, privileged_class)
  # ----------------------------------------------------------------------------
  
  # conversion of targeted variable to numeric
  input_data[[target_variable]] <- as.numeric(as.character(input_data[[target_variable]]))
  
  # Check: ignore protected variable for model explanation
  if(ignore_protected == TRUE){
    # removing protected variable from model explanation calculation
    data <- input_data[colnames(input_data) != protected_variable]
    print('Warning: Protected variable not included in model explanation')
  }
  else
  {
    data <- input_data
  }
  model_explainer <- DALEX::explain(ml_model, data = data[, -1], y = get(target_variable[1], input_data), colorize = FALSE) 
  fairness_obj <- fairness_check(model_explainer, protected = get(protected_variable[1], input_data), privileged = privileged_class)
  
  predictive_equality <- fairness_obj$fairness_check_data$score[3]
  equal_opportunity <- fairness_obj$fairness_check_data$score[4]
  statistical_parity <- fairness_obj$fairness_check_data$score[5]
  
  fairness_score <- list("Predictive Equality" = predictive_equality, 
                         "Equal Opportunity" = equal_opportunity,
                         "Statistical Parity" = statistical_parity)
  
  return(fairness_score)
}
  