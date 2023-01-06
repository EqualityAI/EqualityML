test_that("test resampling bias mitigation method", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1),
    name = as.character(1:10),
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"

  # resampling-uniform
  mitigation_method <- "resampling-uniform"
  data_transformed <- bias_mitigation(mitigation_method, df, target_variable, protected_variable)
  df_1 <-data_transformed$training_data
  expect_equal(nrow(df_1), nrow(df))
  expect_equal(ncol(df_1), ncol(df))
  
  expect_setequal(df_1[[protected_variable]], df[[protected_variable]])
  expect_setequal(df_1[[target_variable]], df[[target_variable]])
  
  
  # resampling-preferential
  probs = c(0.9, 0.82, 0.56, 0.78, 0.45, 0.12, 0.48, 0.63, 0.48, 0.88)
  mitigation_method <- "resampling-preferential"
  data_transformed <- bias_mitigation(mitigation_method, df, target_variable, protected_variable, probs = probs)
  df_2 <-data_transformed$training_data
  
  expect_equal(nrow(df_2), nrow(df))
  expect_equal(ncol(df_2), ncol(df))
  
  expect_setequal(df_2[[protected_variable]], df[[protected_variable]])
  expect_setequal(df_2[[target_variable]], df[[target_variable]])
  
  # test when probs are NULL
  expect_error(bias_mitigation(mitigation_method, df, target_variable, protected_variable))

})

test_that("test reweighing bias mitigation method", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1),
    name = as.character(1:10),
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"
  
  mitigation_method <- "reweighing"
  data_weights <- bias_mitigation(mitigation_method, df, target_variable, protected_variable)
  weights <-data_weights$weights
  expect_length(weights, nrow(df))
})



test_that("test disparate-impact-remover bias mitigation method", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1),
    name = 1:10,
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"
  
  mitigation_method <- "disparate-impact-remover"
  data_transformed <- bias_mitigation(mitigation_method, df, target_variable, protected_variable)
  df_1 <-data_transformed$training_data
  
  expect_equal(nrow(df_1), nrow(df))
  expect_equal(ncol(df_1), ncol(df))
  
  expect_setequal(df_1[[protected_variable]], df[[protected_variable]])
  expect_setequal(df_1[[target_variable]], df[[target_variable]])
  
})


test_that("test incorrect arguments on bias mitigation function", {
  df <- data.frame(
    sex = as.factor(c(rep("M", 5), rep("F", 5))),
    target = c(1, 1, 1, 1, 0, 0, 0, 1, 0, 1),
    name = as.character(1:10),
    stringsAsFactors = FALSE
  )
  target_variable = "target"
  protected_variable = "sex"
  
  expect_error(bias_mitigation("something", df, target_variable, protected_variable))
  expect_error(bias_mitigation("reweighing", c(1,2), target_variable, protected_variable))
  expect_error(bias_mitigation("reweighing", df, "something", protected_variable))
  expect_error(bias_mitigation("reweighing", df, target_variable, "something"))
})