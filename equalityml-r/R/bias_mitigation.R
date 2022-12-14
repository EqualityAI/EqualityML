#' Bias mitigation
#'
#' Bias mitigation is a wrapper with several options to make a dataset more balanced.
#'
#' @param method character, Name of the mitigation method
#'                                    1. "disparate-impact-remover"
#'                                    2. "reweighing"
#'                                    3. "resampling"
#' @param training_data \code{data.frame}, Data to be transformed
#' @param target_variable character, Target variable
#' @param protected_variable character, Data column name which contains sensitive information such as gender, race etc...
#' @param testing_data \code{data.frame}, Data to be transformed
#' @param probs numeric, Vector with probabilities for preferential sampling 
#' @param cutoff numeric, Threshold for probabilities for sampling.  Value from 0 to 1.
#' @param lambda numeric, Amount of repair desired for disparate-impact-remover. 
#' Value from 0 to 1, where 0 will return almost unchanged dataset and 1 fully repaired dataset
#' @param features character, Data columns used to train the model
#' 
#' @return Data/weights after the mitigation (list)
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
#'   training_data = custom_data,
#'   target_variable = "target",
#'   protected_variable = "sex"
#' )
bias_mitigation <- function(method, training_data, target_variable, protected_variable, testing_data = NULL, probs = NULL, lambda = 1, cutoff = 0.5, features = NULL){

  # Check input arguments
  stopifnot(method == "disparate-impact-remover" | method == "reweighing" | method == "resampling" | 
              method == "resampling-uniform"  | method == "resampling-preferential")
  
  if(is.null(features)){
    features = colnames(training_data)
  }
  
  # conversion of protected variable to factor
  training_data[[protected_variable]] <- as.factor(training_data[[protected_variable]])

  # conversion of targeted variable to numeric
  training_data[[target_variable]] <- as.numeric(as.character(training_data[[target_variable]]))

  if(!is.null(testing_data)){
    # conversion of protected variable to factor
    testing_data[[protected_variable]] <- as.factor(testing_data[[protected_variable]])
    
    # conversion of targeted variable to numeric
    testing_data[[target_variable]] <- as.numeric(as.character(testing_data[[target_variable]]))
  }
  
  method <- tolower(method)
  # ----------------------------------------------------------------------------
  if(method == "disparate-impact-remover"){
    training_data_transformed <- disp_removing_data(training_data, features, target_variable, protected_variable, lambda)
    results = list("training_data" = training_data_transformed)
    if(!is.null(testing_data)){
      testing_data_transformed <- disp_removing_data(testing_data, features, target_variable, protected_variable, lambda)
      results <- append(results,list("testing_data" = testing_data_transformed)) 
    }
  }
  else if(method == "reweighing")
  {
    data_weights <- reweighing_model_weights(training_data, target_variable, protected_variable)
    results = list("weights" = data_weights)
  }
  else if((method == "resampling") || (method == "resampling-uniform") || (method == "resampling-preferential"))
  {
    training_data_transformed <- resampling_data(method, training_data, target_variable, protected_variable, probs, cutoff)
    results = list("training_data" = training_data_transformed)  
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
disp_removing_data <- function(input_data, features, target_variable, protected_variable, lambda){

    if (lambda > 1 || lambda < 0) {
        lambda = 1
        print("The lambda value for disparate impact factor has been changed to 1 since lambda>1 or lambda<0")
    }
  
    x <- input_data[features != target_variable]

    # finding list of numeric features
    features_transform <- colnames(x[sapply(x, is.numeric)])
    if(length(features_transform) == 0){
      print("Features to transform are empty")
    }
      

    # Data transformation using Disparate Impact Remover
    data_transformed <- fairmodels::disparate_impact_remover(data = input_data, protected = get(protected_variable[1], input_data),
                                                 features_to_transform = features_transform,
                                                 lambda = lambda)
    return(data_transformed)
}

#' Reweighing Model Weights
#'
#' Function returns weights for model training. The purpose of this weights is to mitigate bias in statistical parity.
#' Obtain weights for machine learning modelusing 'reweight' function from fairmodels package.
#' @noRd
reweighing_model_weights <- function(input_data, target_variable, protected_variable){

    # data weights calculations
    data_weights <- fairmodels::reweight(protected = get(protected_variable[1], input_data), y = get(target_variable[1], input_data))

    return(data_weights)
}

#' Resampling Data
#'
#' Resample the input data using 'resample' function from fairmodels package.
#' @noRd
resampling_data <- function(method, input_data, target_variable, protected_variable, probs = NULL, cutoff = 0.5){

  if (cutoff > 1 || cutoff < 0) {
    cutoff = 0.5
    print("The cutoff value for probabilities threshold has been changed to 0.5 since cutoff>1 or cutoff<0")
  }
  
  if((method == "resampling") || (method == "resampling-uniform")){
    data_index <- fairmodels::resample(protected = get(protected_variable[1], input_data), y = get(target_variable[1], input_data), type = "uniform", cutoff = cutoff)
  }
  else{
    stopifnot(!is.null(probs))
    data_index <- fairmodels::resample(protected = get(protected_variable[1], input_data), y = get(target_variable[1], input_data), type = "preferential", probs = probs, cutoff = cutoff)
  }
  data_transformed <- input_data[data_index,] # mitigated training data
  
  return(data_transformed)
  
}
