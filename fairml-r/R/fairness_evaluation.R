#' Fairness metric
#'
#' Evaluate fairness metrics of a binary classification Machine Learning application.
#'
#' @param ml_model object, Trained machine learning model
#' @param input_data \code{data.frame}, Data for calculating fairness metric
#' @param target_variable character, Target variable
#' @param protected_variable character, Data column name which contains sensitive information such as gender, race etc...
#' @param privileged_class character,  Privileged class from protected variable : "privileged"
#' @param ignore_protected bool, if TRUE, ignore protected variable for model explanation
#'
#'
#' @return Fairness metric score (list)
#' @export
#'
#' @examples
#'
#' set.seed(1)
#' # custom data frame with x1, x2 and y column names
#' custom_data <- data.frame(
#'   x1 = as.factor(c(rep(1, 500), rep(2, 500))),
#'   x2 = c(rnorm(500, 400, 40), rnorm(500, 600, 100))
#'   y = sample(c(0,1), replace=TRUE, size=1000)
#' )
#'
#'ADD ml_model
#'
#' mitigation_result <- bias_mitigation(
#'   ml_model = ml_model,
#'   input_data = custom_data,
#'   target_variable = custom_data$y,
#'   protected_variable = custom_data$x1,
#'   privileged_class = 1
#' )
fairness_metric <- function(ml_model, input_data, target_variable, protected_variable, privileged_class, ignore_protected = TRUE){
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
  
  #Create model explainer object using explain function from DALEX package
  model_explainer <- DALEX::explain(ml_model, data = data[, -1], y = get(target_variable[1], input_data), colorize = FALSE) 
  fairness_obj <- fairness_check(model_explainer, protected = get(protected_variable[1], input_data), privileged = privileged_class)
  
  # Extract required fairness metric scores: predictive_equality, equal_opportunity and statistical_parity
  predictive_equality <- fairness_obj$fairness_check_data$score[3]
  equal_opportunity <- fairness_obj$fairness_check_data$score[4]
  statistical_parity <- fairness_obj$fairness_check_data$score[5]
  
  fairness_score <- list("Predictive Equality" = predictive_equality, 
                         "Equal Opportunity" = equal_opportunity,
                         "Statistical Parity" = statistical_parity)
  
  return(fairness_score)
}
  