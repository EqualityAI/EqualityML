test_that("Test resampling bias mitigation method", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5), rep("N", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1),
    name = as.character(1:15),
    probs = c(0.9, 0.82, 0.56, 0.78, 0.45, 0.12, 0.48, 0.63, 0.48, 0.88, 0.34, 0.12, 0.34, 0.49, 0.9),
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"
  
  # resampling
  MN <- sum(df[[protected_variable]]  == "M" & df[[target_variable]] == 0)
  MP <- sum(df[[protected_variable]]  == "M" & df[[target_variable]] == 1)
  FN <- sum(df[[protected_variable]]  == "F" & df[[target_variable]] == 0)
  FP <- sum(df[[protected_variable]]  == "F" & df[[target_variable]] == 1)
  NN <- sum(df[[protected_variable]]  == "N" & df[[target_variable]] == 0)
  NP <- sum(df[[protected_variable]]  == "N" & df[[target_variable]] == 1)
  
  weights <- fairmodels::reweight(df[[protected_variable]] , df[[target_variable]])
  
  wMP <- weights[1]
  wMN <- weights[5]
  wFP <- weights[10]
  wFN <- weights[6]
  wNN <- weights[13]
  wNP <- weights[15]
  
  # expected
  E_MP <- round(MP * wMP)
  E_MN <- round(MN * wMN)
  E_FN <- round(FN * wFN)
  E_FP <- round(FP * wFP)
  E_NP <- round(NP * wNP)
  E_NN <- round(NN * wNN)
  
  # 1 - resampling-uniform
  mitigation_method <- "resampling-uniform"
  data_transformed <- bias_mitigation(mitigation_method, df, target_variable, protected_variable)
  df_1 <-data_transformed$data
  
  MN_1 <- sum(df_1[[protected_variable]] == "M" & df_1[[target_variable]] == 0)
  MP_1 <- sum(df_1[[protected_variable]]  == "M" & df_1[[target_variable]] == 1)
  FN_1 <- sum(df_1[[protected_variable]]  == "F" & df_1[[target_variable]] == 0)
  FP_1 <- sum(df_1[[protected_variable]]  == "F" & df_1[[target_variable]] == 1)
  NN_1 <- sum(df_1[[protected_variable]]  == "N" & df_1[[target_variable]] == 0)
  NP_1 <- sum(df_1[[protected_variable]]  == "N" & df_1[[target_variable]] == 1)
  
  expect_equal(E_MP, MP_1)
  expect_equal(E_MN, MN_1)
  expect_equal(E_FP, FP_1)
  expect_equal(E_FN, FN_1)
  expect_equal(E_NP, MP_1)
  expect_equal(E_NN, NN_1)
  
  # 2 - resampling-preferential
  mitigation_method <- "resampling-preferential"
  data_transformed <- bias_mitigation(mitigation_method, df, target_variable, protected_variable, probs = df$probs)
  df_2 <-data_transformed$data
  
  expect_equal(sort(as.numeric(df_2$name)), c(1, 2, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 15))
  
  expect_error(bias_mitigation(mitigation_method, df, target_variable, protected_variable))
  expect_error(bias_mitigation(mitigation_method, df, target_variable, protected_variable, probs = df$probs, cutoff = 12))
  expect_error(bias_mitigation(mitigation_method, df, target_variable, protected_variable, probs = df$probs, cutoff = c(0.3, 0.4)))
  
})

test_that("Test reweighing bias mitigation method", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5), rep("N", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1),
    name = as.character(1:15),
    probs = c(0.9, 0.82, 0.56, 0.78, 0.45, 0.12, 0.48, 0.63, 0.48, 0.88, 0.34, 0.12, 0.34, 0.49, 0.9),
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"
  
  # 3 - reweighing
  mitigation_method <- "reweighing"
  data_weights <- bias_mitigation(mitigation_method, df, target_variable, protected_variable)
  weights <-data_weights$weights

})



test_that("Test disparate-impact-remover bias mitigation method", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5), rep("N", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1),
    name = as.character(1:15),
    probs = c(0.9, 0.82, 0.56, 0.78, 0.45, 0.12, 0.48, 0.63, 0.48, 0.88, 0.34, 0.12, 0.34, 0.49, 0.9),
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"
  
  
  # 4 - disparate-impact-remover
  mitigation_method <- "disparate-impact-remover"
  data_transformed <- bias_mitigation(mitigation_method, df, target_variable, protected_variable)
  
})