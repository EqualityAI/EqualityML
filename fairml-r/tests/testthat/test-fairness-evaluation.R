test_that("Test fairness metric evaluation", {
  df <- data.frame(
    sex = c(rep(2, 10), rep(1, 10), rep(0, 10)),
    age = 1:30,
    target = c(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1)
  )
  
  df <- data.frame(
    sex = c(rep(2, 100), rep(1, 100), rep(0, 100)),
    age = c(rep(1:30,10)),
    target = c(rep(c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),15))
  )
  
  print(df)
  target_variable = "target"
  protected_variable = "sex"
  ml_model <- glm(target ~ sex + age, data = df, family = 'binomial')
  
  fairness_score <- fairness_metric(ml_model = ml_model, input_data = df, 
                                    target_variable = target_variable, 
                                    protected_variable = protected_variable,
                                    privileged_class = 2,
                                    ignore_protected = FALSE)
  print(fairness_score)
})