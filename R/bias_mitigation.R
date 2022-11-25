#===============================================================================
# MITIGATION METHOD
#===============================================================================
bias_mitigation <- function(method, data_input, target_var, param_bias_mitigation){
  # Applying mitigation 
  #
  # INPUT
  # method (character)          : Name of the mitigation method
  #                                 1. Disparate Impact Remove: "disparate-impact-remover"
  #                                 2. Reweighting : 'reweight'
  #                                 3. Resampling : 'resample'
  # data_input (list-data.frame): Input data
  # target_var (character)      : Target variable 
  # param_bias_mitigation (list): Parameters for the mitigation method
  #                                 1. 'protected_var' - Protected variable
  #
  # OUTPUT
  # results (list)              : Data/indices/weight after the mitigation 
  #
  # EXAMPPLE
  # training_data_m <- bias_mitigation(mitigation_method, data_clean$training, target_var, param_bias_mitigation)
  # ----------------------------------------------------------------------------
  protected_var <- param_bias_mitigation$protected
  # conversion of protected variable to factor
  data_input[[protected_var]] <- as.factor(data_input[[protected_var]])
  # conversion of targeted variable to numeric
  data_input[[target_var]] <- as.numeric(as.character(data_input[[target_var]]))
  # ----------------------------------------------------------------------------
  if((tolower(method) == "disparate-impact-remover") | (tolower(method) == "disparate impact remover")){
    if(!'lambda' %in% names(param_bias_mitigation)){
      lambda <- 1
    }else if (!is.numeric(lambda)){
      lambda <- 1
    }
    if (lambda>1||lambda<0) {lambda=1; print("The lambda value for disparate impact factor has been changed to 1 since lambda>1 or lambda<0")}
    data_transformed <- mitigation_data_transform(method, data_input, target_var, protected_var,lambda)
    results = list("data" = data_transformed$data)
  }
  else if((tolower(method) == "reweight") | (tolower(method) == "reweighting")){
    model_reweight <- mitigation_data_weights(method, data_input, target_var, protected_var)
    results = list("weight" = model_reweight$weight)
  }
  else if((tolower(method) == "resample") | (tolower(method) == "resampling")){
    index_resample <- mitigation_data_weights(method, data_input, target_var, protected_var)
    results = list("index" = index_resample$index)
  } else{
    print('Mitigation Method - Invalid/Not Available')
  }
  return(results)
}
#------------------------------------------------------------------------------
# METHODS - TRANSFORM DATASET
#------------------------------------------------------------------------------
mitigation_data_transform <- function(method, data_input, target_var, protected_var, lambda){
  # mitigate bias with disparate impact remover pre-processing method (Feldman et al. (2015))
  if((tolower(method) == "disparate-impact-remover") | (tolower(method) == "disparate impact remover")){
    # finding list of numeric features
    features_transform <- colnames(data_input[sapply(data_input, is.numeric)])
    # removing target variable from the features transform list
    features_transform <- features_transform[features_transform != target_var[1]]
    # Data transformation using Disparate Impact Remover
    data_transformed <- disparate_impact_remover(data = data_input, protected = get(protected_var[1], data_input), 
                                                  features_to_transform = features_transform,
                                                 lambda=lambda)
    data_transformed = list("data" = data_transformed)
    return(data_transformed)
  }
}
#------------------------------------------------------------------------------
# METHODS - DATA SAMPLE WEIGHTS/ RESAMPLE INDICES
#------------------------------------------------------------------------------
mitigation_data_weights <- function(method, data_input, target_var, protected_var){
  if((tolower(method) == "reweight") | (tolower(method) == "reweighting")){
    # Check
    # finding list of numeric features
    #features_transform <- colnames(data_input[sapply(data_input, is.numeric)])
    # removing target variable from the features transform list
    #features_transform <- features_transform[features_transform != target_var[1]]
    
    # Delete
    # conversion of targeted variable to numeric
    #data_input[[target_var]] <- as.numeric(as.character(data_input[[target_var]]))
    # conversion of protected variable to factor
    #data_input[[protected_var]] <- as.factor(data_input[[protected_var]])
    
    # data weights calculations
    data_weights <- fairmodels::reweight(protected = get(protected_var[1], data_input), y = get(target_var[1], data_input)) 
    data_weights = list("weight" = data_weights)
    return(data_weights)
  } 
  else if((tolower(method) == "resample") | (tolower(method) == "resampling")){
    
    # Delete
    # conversion of targeted variable to numeric
    #data_input[[target_var]] <- as.numeric(as.character(data_input[[target_var]]))
    # conversion of protected variable to factor
    #data_input[[protected_var]] <- as.factor(data_input[[protected_var]])
    
    # data resample index
    data_index <- fairmodels::resample(protected = get(protected_var[1], data_input), y = get(target_var[1], data_input))
    data_index = list("index" = data_index)
    return(data_index)
  }
}