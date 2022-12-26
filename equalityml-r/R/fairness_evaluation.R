#' Fairness metric
#'
#' Evaluate fairness metrics of a binary classification Machine Learning application.
#'
#' @param ml_model object, Trained machine learning model
#' @param input_data \code{data.frame}, Data for calculating fairness metric
#' @param target_variable character, Target variable
#' @param protected_variable character, Data column name which contains sensitive information such as gender, race etc...
#' @param privileged_class character,  Privileged class from protected variable : "privileged"
#' @param features character, Data columns 
#'
#'
#' @return Fairness metric score (list)
#' @export
#'
#' @examples
#'
#' set.seed(1)
#' # custom data frame with sex, age and target column names
#' custom_data <- data.frame(
#'   sex = c(rep("M", 140), rep("F", 60)),
#'   age = c(rep(1:20,10)),
#'   target = c(
#'   c(rep(c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),14)),
#'   c(rep(c(0, 1, 0, 1, 0, 0, 1, 0, 0, 1),6))
#'   )
#' )
#'
#'ml_model <- glm(target ~ sex + age, data = custom_data, family = 'binomial')
#'
#' fairness_score <- fairness_metric(
#'   ml_model = ml_model,
#'   input_data = custom_data,
#'   target_variable = "target",
#'   protected_variable = "sex",
#'   privileged_class = "M"
#' )
fairness_metric <- function(ml_model, input_data, target_variable, protected_variable, privileged_class, features = NULL){
  # conversion of targeted variable to numeric
  input_data[[target_variable]] <- as.numeric(as.character(input_data[[target_variable]]))
  
  if(is.null(features)){
    features = colnames(input_data)
  }
  
  # Get x data 
  x <- input_data[features != target_variable]
  
  #Create model explainer object using explain function from DALEX package
  model_explainer <- DALEX::explain(ml_model, data = x, y = get(target_variable[1], input_data), colorize = FALSE, verbose = FALSE) 
  fairness_obj <- fairmodels::fairness_check(model_explainer, protected = get(protected_variable[1], input_data), privileged = privileged_class, verbose = FALSE)
  
  # Extract required fairness metric scores: predictive_equality, equal_opportunity and statistical_parity
  predictive_equality <- fairness_obj$fairness_check_data$score[3]
  equal_opportunity <- fairness_obj$fairness_check_data$score[4]
  statistical_parity <- fairness_obj$fairness_check_data$score[5]
  
  fairness_score <- list("Predictive Equality" = predictive_equality, 
                         "Equal Opportunity" = equal_opportunity,
                         "Statistical Parity" = statistical_parity)
  return(fairness_score)
}
  