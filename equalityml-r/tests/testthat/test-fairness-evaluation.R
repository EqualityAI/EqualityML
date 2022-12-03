test_that("test fairness metric evaluation", {
  df <- data.frame(
    sex = c(rep("M", 140), rep("F", 60)),
    age = c(rep(1:20,10)),
    target = c(
      c(rep(c(1, 1, 1, 1, 1, 1, 1, 0, 0, 0),14)),
      c(rep(c(0, 1, 0, 1, 0, 0, 1, 0, 0, 1),6))
    )
  )
  
  target_variable = "target"
  protected_variable = "sex"
  ml_model <- glm(target ~ sex + age, data = df, family = 'binomial')
  
  fairness_score <- fairness_metric(ml_model = ml_model, input_data = df, 
                                    target_variable = target_variable, 
                                    protected_variable = protected_variable,
                                    privileged_class = "M",
                                    ignore_protected = FALSE)
  
  expect_equal(fairness_score$'Predictive Equality', 0.4)
  expect_equal(fairness_score$'Equal Opportunity', 0.25)
  expect_equal(fairness_score$'Statistical Parity', 0.315, tolerance=1e-2)
  
})