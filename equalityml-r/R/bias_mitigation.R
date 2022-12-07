#' Bias mitigation
#'
#' Bias mitigation is a wrapper with several options to make a dataset more balanced.
#'
#' @param method character, Name of the mitigation method
#'                                    1. "disparate-impact-remover"
#'                                    2. "reweighing"
#'                                    3. "resampling"
#' @param data_input \code{data.frame}, data to be transformed
#' @param target_variable character, target variable
#' @param protected_variable character, data column name which contains sensitive information such as gender, race etc...
#' @param probs numeric, vector with probabilities for preferential sampling 
#' @param cutoff numeric, threshold for probabilities for sampling.  Value from 0 to 1.
#' @param lambda numeric, amount of repair desired for disparate-impact-remover. 
#' Value from 0 to 1, where 0 will return almost unchanged dataset and 1 fully repaired dataset
#'
#' @return Data/indices/weights after the mitigation (list)
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
#' mitigation_result <- bias_mitigation(
#'   method = "disparate-impact-remover",
#'   data_input = custom_data,
#'   target_variable = "target",
#'   protected_variable = "sex"
#' )
bias_mitigation <- function(method, data_input, target_variable, protected_variable, probs = NULL, lambda = 1, cutoff = 0.5){

  # Check input arguments
  stopifnot(method == "disparate-impact-remover" | method == "reweighing" | method == "resampling" | 
              method == "resampling-uniform"  | method == "resampling-preferential")
  
  # conversion of protected variable to factor
  data_input[[protected_variable]] <- as.factor(data_input[[protected_variable]])

  # conversion of targeted variable to numeric
  data_input[[target_variable]] <- as.numeric(as.character(data_input[[target_variable]]))

  method <- tolower(method)
  # ----------------------------------------------------------------------------
  if(method == "disparate-impact-remover"){
    results <- disp_removing_data(data_input, target_variable, protected_variable, lambda)
  }
  else if(method == "reweighing")
  {
    results <- reweighing_model_weights(data_input, target_variable, protected_variable)
  }
  else if((method == "resampling") || (method == "resampling-uniform") || (method == "resampling-preferential"))
  {
    results <- resampling_data(method, data_input, target_variable, protected_variable, probs, cutoff)
  } else{
    print('Mitigation Method - Invalid/Not Available')
  }
  return(results)
}

#' Disp Removing Data
#'
#' This function mitigates bias with disparate impact remover pre-processing method (Feldman et al. (2015))
#' Filters out sensitive correlations in a dataset using 'disparate_impact_remover' function from fairmodels package.
#' @noRd
disp_removing_data <- function(data_input, target_variable, protected_variable, lambda){

    if (lambda > 1 || lambda < 0) {
        lambda = 1
        print("The lambda value for disparate impact factor has been changed to 1 since lambda>1 or lambda<0")
    }

    # finding list of numeric features
    features_transform <- colnames(data_input[sapply(data_input, is.numeric)])
    
    # removing target variable from the features transform list
    features_transform <- features_transform[features_transform != target_variable[1]]
    if(length(features_transform) == 0){
      print("Features to transform are empty")
    }
      

    # Data transformation using Disparate Impact Remover
    data_transformed <- fairmodels::disparate_impact_remover(data = data_input, protected = get(protected_variable[1], data_input),
                                                 features_to_transform = features_transform,
                                                 lambda = lambda)
    data_transformed = list("data" = data_transformed)
    return(data_transformed)
}

#' Reweighing Model Weights
#'
#' Function returns weights for model training. The purpose of this weights is to mitigate bias in statistical parity.
#' Obtain weights for machine learning modelusing 'reweight' function from fairmodels package.
#' @noRd
reweighing_model_weights <- function(data_input, target_variable, protected_variable){

    # data weights calculations
    data_weights <- fairmodels::reweight(protected = get(protected_variable[1], data_input), y = get(target_variable[1], data_input))
    data_weights = list("weights" = data_weights)

    return(data_weights)
}

#' Resampling Data
#'
#' Resample the input data using 'resample' function from fairmodels package.
#' @noRd
resampling_data <- function(method, data_input, target_variable, protected_variable, probs = NULL, cutoff = 0.5){

  if (cutoff > 1 || cutoff < 0) {
    cutoff = 0.5
    print("The cutoff value for probabilities threshold has been changed to 0.5 since cutoff>1 or cutoff<0")
  }
  
  if((method == "resampling") || (method == "resampling-uniform")){
    data_index <- fairmodels::resample(protected = get(protected_variable[1], data_input), y = get(target_variable[1], data_input), type = "uniform", cutoff = cutoff)
  }
  else{
    stopifnot(!is.null(probs))
    data_index <- fairmodels::resample(protected = get(protected_variable[1], data_input), y = get(target_variable[1], data_input), type = "preferential", probs = probs, cutoff = cutoff)
  }
  data_transformed <- data_input[data_index,] # mitigated training data
  data_transformed = list("data" = data_transformed)
  
  return(data_transformed)
  
}