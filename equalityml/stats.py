import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from scipy import stats
from sklearn.model_selection import train_test_split
from equalityml.threshold import binary_threshold_score, discrimination_threshold, DECISION_MAKER


def paired_ttest(model_1,
                 X,
                 y,
                 model_2=None,
                 method="mcnemar",
                 threshold=0.5,
                 fair_object=None,
                 mitigation_method=None,
                 scoring=None,
                 utility_costs=None,
                 compute_discrimination_threshold=False,
                 decision_maker=DECISION_MAKER,
                 random_seed=None):
    """
    Statistical paired t test for classifier comparisons. 2 methods are provided: McNemar's test and paired ttest 5x2cv.

    Parameters
    ----------
    model_1 : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values. The target y must be a binary classification target.
    model_2 : a Scikit-Learn estimator, default=None
        A scikit-learn estimator that should be a classifier. For 5x2cv model it can be None when a mitigation method
        is provided together with a FAIR object.
    method : str, default="mcnemar"
        If `mcnemar` uses McNemar's test, if "5x2cv" uses paired ttest 5x2cv.
    threshold : float, default=0.5
        Discrimination threshold for predicting the favorable class.
    fair_object: FAIR, default=None
        FAIR object with methods to apply bias mitigation and evaluate fairness metric.
    mitigation_method : str, default=None
         Name of the bias mitigation method to reduce unfairness of the Machine Learning model using `fair_object`.
         Accepted values:
         "resampling"
         "resampling-preferential"
         "reweighing"
         "disparate-impact-remover"
         "correlation-remover"
    scoring : str, callable. Default=None
        If None (default), uses 'accuracy' for sklearn classifiers
        If 'cost', uses utility_costs parameter to calculate the score
        If str, uses a sklearn scoring metric string identifier, for example {accuracy, f1, precision, recall, roc_auc}.
        If a callable object or function is provided, it has to agree with sklearn's signature 'scorer(estimator, X, y)'.
        Check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html for more information.
    utility_costs : list, default=None
        Utility costs for cost-sensitive learning. It has to be a 4 element list where the cost values correspond to the
        following cost sequence: [TP, FN, FP, TN]
    compute_discrimination_threshold : bool, default=True,
        Compute the best discrimination threshold for each mitigated model based on the input decision maker.
    decision_maker : tuple, default=('f1', 'max')
        The metric and decision to optimize the discrimination threshold. The metric shall be available in input metrics
        list and 3 different decisions can be used: 'max', 'min' or 'limit' with the following behaviour:
        - 'max' computes the threshold which maximizes the selected metric
        - 'min' computes the threshold which minimizes the selected metric
        - 'limit' requires an extra float parameter between 0 and 1. The optimal threshold is calculated when the
        selected metric reaches that limit.
    random_seed : int, default=None
        Random seed for creating the test/train splits.
    Returns
    ----------
    chi2 : float
        Chi-squared value
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level (e.g 0.05) is larger than the p-value, we reject the null hypothesis and accept that
        there are significant differences in the two compared models.
    """

    if method.lower() == "mcnemar":
        chi2, p = mcnemar(model_1,
                          model_2,
                          X,
                          y,
                          threshold=threshold)
    elif method == "5x2cv":
        chi2, p = paired_ttest_5x2cv(model_1,
                                     X,
                                     y,
                                     model_2=model_2,
                                     threshold=threshold,
                                     fair_object=fair_object,
                                     mitigation_method=mitigation_method,
                                     scoring=scoring,
                                     utility_costs=utility_costs,
                                     compute_discrimination_threshold=compute_discrimination_threshold,
                                     decision_maker=decision_maker,
                                     random_seed=random_seed)
    else:
        print(f"Invalid method {method}")
        chi2, p = (0, 1)

    return chi2, p


def mcnemar_table(model_1,
                  model_2,
                  X,
                  y,
                  threshold=0.5):
    """
    Compute a 2x2 contigency table for McNemar's test.
    Parameters
    -----------
    model_1 : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.
    model_2 : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    threshold : float, default=0.5
        Discrimination threshold for predicting the favorable class.
    Returns
    ----------
    tb : array-like, shape=[2, 2]
       2x2 contingency table with the following contents:
       a: tb[0, 0]: # of samples that both models predicted correctly
       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
       d: tb[1, 1]: # of samples that both models predicted incorrectly
    """

    if model_2 is None:
        raise ValueError("For McNemar method, model 2 cannot be None")

    if getattr(model_1, "_estimator_type", None) != "classifier":
        raise TypeError("Model has to be a classifier")

    if getattr(model_2, "_estimator_type", None) != "classifier":
        raise TypeError("Model has to be a classifier")

    # Compute predictions for model 1 and 2
    y_model1 = list(map(lambda x: 1 if x > threshold else 0, model_1.predict_proba(X)[:, -1]))
    y_model2 = list(map(lambda x: 1 if x > threshold else 0, model_2.predict_proba(X)[:, -1]))
    model1_true = (y == y_model1)
    model2_true = (y == y_model2)

    tb = np.zeros((2, 2), dtype=int)
    tb[0, 0] = np.sum((model1_true == 1) & (model2_true == 1))
    tb[0, 1] = np.sum((model1_true == 1) & (model2_true == 0))
    tb[1, 0] = np.sum((model1_true == 0) & (model2_true == 1))
    tb[1, 1] = np.sum((model1_true == 0) & (model2_true == 0))

    return tb


def mcnemar(model_1,
            model_2,
            X,
            y,
            threshold=0.5,
            corrected=True,
            exact_binomial_test=False):
    """
    McNemar's test used on paired nominal data.
    Parameters
    -----------
    model_1 : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier.
    model_2 : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier.
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    threshold : float, default=0.5
        Discrimination threshold for predicting the favorable class.
    corrected : bool, default=True
        If `True`, uses Edward's continuity correction for chi-squared
    exact_binomial_test : bool, default=False
        If `True`, uses an exact binomial test comparing b to
        a binomial distribution with n = b + c and p = 0.5.
        It is highly recommended to use `exact_binomial_test=True` for sample sizes < 25
        since chi-squared is not well-approximated by the chi-squared distribution.
    Returns
    ----------
    chi2 : float
        Chi-squared value
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level (e.g 0.05) is larger than the p-value, we reject the null hypothesis and accept that
        there are significant differences in the two compared models.
    """

    tb = mcnemar_table(model_1, model_2, X, y, threshold=threshold)

    b = tb[0, 1]
    c = tb[1, 0]

    if exact_binomial_test is False:
        if corrected:
            chi2 = (abs(b - c) - 1.0) ** 2 / float(b + c)
        else:
            chi2 = (b - c) ** 2 / float(b + c)
        p = stats.distributions.chi2.sf(chi2, 1)

    else:
        chi2 = None
        p = min(stats.binom.cdf(min(b, c), b + c, 0.5) * 2.0, 1.0)

    return chi2, p


def paired_ttest_5x2cv(model_1,
                       X,
                       y,
                       model_2=None,
                       threshold=0.5,
                       fair_object=None,
                       mitigation_method=None,
                       scoring=None,
                       utility_costs=None,
                       compute_discrimination_threshold=False,
                       decision_maker=DECISION_MAKER,
                       random_seed=None):
    """
    Implements the 5x2cv paired t test for comparing the performance of two models (classifier or regressors).
    This test was proposed by Dieterrich (1998).
    Note: the second model can be computed by applying the input mitigation method to the first model.
    Parameters
    ----------
    model_1 : a Scikit-Learn estimator
        A scikit-learn estimator that can be a classifier or regressor.
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    model_2 : a Scikit-Learn estimator, default=None
        A scikit-learn estimator that should be a classifier. It can be None when a mitigation method is provided
        together with a FAIR object.
    threshold : float, default=0.5
        Discrimination threshold for predicting the favorable class.
    fair_object: FAIR, default=None
        FAIR object with methods to apply bias mitigation and evaluate fairness metric.
    mitigation_method : str, default=None
         Name of the bias mitigation method to reduce unfairness of the Machine Learning model using `fair_object`.
         Accepted values:
         "resampling"
         "resampling-preferential"
         "reweighing"
         "disparate-impact-remover"
         "correlation-remover"
    scoring : str, callable. Default=None
        If None (default), uses 'accuracy' for sklearn classifiers
        If 'cost', uses utility_costs parameter to calculate the score
        If str, uses a sklearn scoring metric string identifier, for example {accuracy, f1, precision, recall, roc_auc}.
        If a callable object or function is provided, it has to agree with sklearn's signature 'scorer(estimator, X, y)'.
        Check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html for more information.
    utility_costs : list, default=None
        Utility costs for cost-sensitive learning. It has to be a 4 element list where the cost values correspond to the
        following cost sequence: [TP, FN, FP, TN]
    compute_discrimination_threshold : bool, default=True,
        Compute the best discrimination threshold for each mitigated model based on the input decision maker.
    decision_maker : tuple, default=('f1', 'max')
        The metric and decision to optimize the discrimination threshold. The metric shall be available in input metrics
        list and 3 different decisions can be used: 'max', 'min' or 'limit' with the following behaviour:
        - 'max' computes the threshold which maximizes the selected metric
        - 'min' computes the threshold which minimizes the selected metric
        - 'limit' requires an extra float parameter between 0 and 1. The optimal threshold is calculated when the
        selected metric reaches that limit.
    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.
    Returns
    ----------
    chi2 : float
        Chi-squared value
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level (e.g 0.05) is larger than the p-value, we reject the null hypothesis and accept that
        there are significant differences in the two compared models.
    """
    rng = np.random.RandomState(random_seed)
    # Get scoring when is None
    if scoring is None:
        if model_1._estimator_type == "classifier":
            scoring = "accuracy"
        elif model_1._estimator_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Model must be a Classifier or Regressor.")

    # Check if model 2 will be used or mitigation must be done
    apply_mitigation = False
    if model_2 is None:
        apply_mitigation = True
        if fair_object is None or mitigation_method is None:
            raise ValueError("Model 2 and 'fair_object' or mitigation method cannot be all None")
    else:
        if mitigation_method is not None:
            apply_mitigation = True
            print("Model 2 will be ignored. Mitigation method will be applied.")

    # Get scorer call function
    if fair_object is not None and scoring in fair_object.fairness_metrics_list:
        def scorer(model, X, y):
            # Compute discrimination threshold
            if compute_discrimination_threshold:
                fair_object.threshold = discrimination_threshold(model,
                                                                 X,
                                                                 y,
                                                                 fair_object=fair_object,
                                                                 decision_maker=decision_maker,
                                                                 metrics=[decision_maker[0]],
                                                                 utility_costs=utility_costs,
                                                                 show=False,
                                                                 model_training=False,
                                                                 random_seed=random_seed)
            else:
                fair_object.threshold = threshold
            fair_object.mitigated_testing_data = pd.concat([X, y], axis=1)
            fair_object.ml_model = model
            return fair_object.fairness_metric(scoring)
    else:
        def scorer(model, X, y):
            # Compute discrimination threshold
            if compute_discrimination_threshold:
                dt = discrimination_threshold(model,
                                              X,
                                              y,
                                              decision_maker=decision_maker,
                                              metrics=[decision_maker[0]],
                                              utility_costs=utility_costs,
                                              show=False,
                                              model_training=False,
                                              random_seed=random_seed)
            else:
                dt = threshold
            return binary_threshold_score(model,
                                          X,
                                          y,
                                          scoring=scoring,
                                          threshold=dt,
                                          utility_costs=utility_costs)

    def _score_diff(_model_1, _model_2, _X_train, _y_train, _X_test, _y_test):
        """ Compute score difference between model 1 and mitigated model"""

        # Train model 1 and get score 1
        _model_1.fit(_X_train, _y_train)
        score_1 = scorer(_model_1, _X_test, _y_test)

        if apply_mitigation or fair_object is not None and scoring in fair_object.fairness_metrics_list:
            # Update training/testing data and reference ml model
            fair_object.training_data = pd.concat([_X_train, _y_train], axis=1)
            fair_object.testing_data = pd.concat([_X_test, _y_test], axis=1)
            fair_object.update_classifier(_model_1)

            if apply_mitigation:
                # Apply bias mitigation to get mitigated model 2
                _model_2 = fair_object.model_mitigation(mitigation_method=mitigation_method)

            if fair_object.mitigated_testing_data is not None:
                testing_data = fair_object.mitigated_testing_data
            else:
                testing_data = fair_object.testing_data

            # Get score 2
            _mitigated_X_test = testing_data.drop(columns=fair_object.target_variable)
            _mitigated_y_test = testing_data[fair_object.target_variable]
            score_2 = scorer(_model_2, _mitigated_X_test, _mitigated_y_test)
        else:
            # Train model 2 and get score 2
            _model_2.fit(_X_train, _y_train)
            score_2 = scorer(_model_2, _X_test, _y_test)

        score_diff = score_1 - score_2
        return score_diff

    sum_variance = 0.0
    first_score_diff = None

    _model_1 = copy.deepcopy(model_1)
    _model_2 = copy.deepcopy(model_2)

    for _ in tqdm(range(5)):

        randint = rng.randint(low=0, high=32768)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=randint)

        score_diff_A = _score_diff(_model_1, _model_2, X_train, y_train, X_test, y_test)
        score_diff_B = _score_diff(_model_1, _model_2, X_test, y_test, X_train, y_train)
        mean_diff = (score_diff_A + score_diff_B) / 2.0
        var_diff = (score_diff_A - mean_diff) ** 2 + (score_diff_B - mean_diff) ** 2
        sum_variance += var_diff

        if first_score_diff is None:
            first_score_diff = score_diff_A

    if abs(first_score_diff) < 1e-100 or sum_variance < 1e-100:
        print("No relevant difference exists between model 1 and model 2")
        return 0, 1

    t_stat = first_score_diff / (np.sqrt(1 / 5.0 * sum_variance))
    p_value = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return float(t_stat), float(p_value)
