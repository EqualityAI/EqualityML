import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from fair import FAIR


## TODO Add discrimination threshold computation


def compare_models(model_1, model_2, data, target_variable, fair_object=None, score="binary_classification",  scoring=None, random_seed=None):
    X = data.drop(columns=target_variable)
    y = data[target_variable]

    if score == "binary_classification":
        # model_1 can have the same architecture as model_2
        tb = mcnemar_table(model_1, model_2, X, y)
        if tb[0, 1] + tb[1, 0] > 25:
            chi2, p = mcnemar(tb)
        else:
            chi2, p = mcnemar(tb, exact_binomial_test=True)
    elif score == "probability":
        # model_1 can not have the same architecture as model_2
        mitigation_method = "reweighing"
        chi2, p = paired_ttest_5x2cv(model_1, model_2, X, y, fair_object, mitigation_method, scoring=scoring, random_seed=random_seed)
        # Fairness metric
        #-chi2, p = paired_ttest_5x2cv(model_1, model_2, X, y, fair_object, mitigation_method, scoring=scoring, random_seed=random_seed)
    return chi2, p


def mcnemar_table(model_1, model_2, X, y):
    """
    Compute a 2x2 contigency table for McNemar's test.
    Parameters
    -----------
    model_1 : object
        Trained Machine Learning model 1 object
    model_2 : object
        Trained Machine Learning model 2 object
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    Returns
    ----------
    tb : array-like, shape=[2, 2]
       2x2 contingency table with the following contents:
       a: tb[0, 0]: # of samples that both models predicted correctly
       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
       d: tb[1, 1]: # of samples that both models predicted incorrectly
    """

    # Compute predictions for model 1 and 2
    ## TODO add discrimination Threshold
    y_model1 = model_1.predict(X)
    y_model2 = model_2.predict(X)

    model1_true = (y == y_model1)
    model2_true = (y == y_model2)

    tb = np.zeros((2, 2), dtype=int)
    tb[0, 0] = np.sum((model1_true == 1) & (model2_true == 1))
    tb[0, 1] = np.sum((model1_true == 1) & (model2_true == 0))
    tb[1, 0] = np.sum((model1_true == 0) & (model2_true == 1))
    tb[1, 1] = np.sum((model1_true == 0) & (model2_true == 0))

    return tb


def mcnemar(tb, corrected=True, exact_binomial_test=False):
    """
    McNemar's test used on paired nominal data.
    Parameters
    -----------
    tb : array-like, shape=[2, 2]
        2 x 2 contigency table (as returned by mcnemar_table),
        where
        a: tb[0, 0]: # of samples that both models predicted correctly
        b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
        c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
        d: tb[1, 1]: # of samples that both models predicted incorrectly
    corrected : bool (default: True)
        True to use Edward's continuity correction for chi-squared
    exact_binomial_test : bool, (default: False)
        If `True`, uses an exact binomial test comparing b to
        a binomial distribution with n = b + c and p = 0.5.
        It is highly recommended to use `exact=True` for sample sizes < 25
        since chi-squared is not well-approximated by the chi-squared distribution.
    Returns
    -----------
    chi2, p : float or None, float
        Returns the chi-squared value and the p-value;
        if `exact_binomial_test=True`, `chi2` is `None`
    """

    if tb.shape != (2, 2):
        raise ValueError("Input array must be a 2x2 array.")

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


def paired_ttest_5x2cv(model_1, model_2, X, y, fair_object, mitigation_method, scoring=None, random_seed=None):
    """
    Implements the 5x2cv paired t test for comparing the performance of two models (classifier or regressors).
    This test was proposed by Dieterrich (1998).
    Parameters
    ----------
    model_1 : scikit-learn classifier or regressor
    model_2 : scikit-learn classifier or regressor
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.
    Returns
    ----------
    t : float
        The t-statistic
    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.
    """
    rng = np.random.RandomState(random_seed)

    if model_1._estimator_type != model_2._estimator_type:
        raise AttributeError("Models must be of the same type")

    if scoring is None:
        if model_1._estimator_type == "classifier":
            scoring = "accuracy"
        elif model_1._estimator_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Model must be a Classifier or Regressor.")

    ## TODO add fairness_metric call
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    elif scoring == "fairness_metric":
        scorer = scoring
    else:
        scorer = scoring

    def _score_diff(_model_1, _model_2, _X_train, _y_train, _X_test, _y_test):

        _model_1.fit(_X_train, _y_train)
        score_1 = scorer(_model_1, _X_test, _y_test)

        # apply bias mitigation to the new dataset
        fair_object.training_data = pd.concat([_X_train, _y_train], axis=1)
        fair_object.testing_data = pd.concat([_X_test, _y_test], axis=1)
        fair_object.update_classifier(_model_1)
        _model_2 = fair_object.mitigate_model(mitigation_method=mitigation_method)

        _mitigated_X_test = fair_object.mitigated_testing_data.drop(columns=fair_object.target_variable)
        _mitigated_y_test = fair_object.mitigated_testing_data[fair_object.target_variable]
        score_2 = scorer(_model_2, _mitigated_X_test, _mitigated_y_test)
        score_diff = score_1 - score_2
        return score_diff

    sum_variance = 0.0
    first_score_diff = None
    for i in range(5):

        randint = rng.randint(low=0, high=32768)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=randint)

        score_diff_A = _score_diff(model_1, model_2, X_train, y_train, X_test, y_test)
        score_diff_B = _score_diff(model_1, model_2, X_test, y_test, X_train, y_train)
        mean_diff = (score_diff_A + score_diff_B) / 2.0
        var_diff = (score_diff_A - mean_diff) ** 2 + (score_diff_B - mean_diff) ** 2
        sum_variance += var_diff

        if first_score_diff is None:
            first_score_diff = score_diff_A

    if abs(first_score_diff) < 1e-10 or sum_variance < 1e-10:
        print("No relevant difference exists between model 1 and model 2")
        return 0, 1

    t_stat = first_score_diff / (np.sqrt(1 / 5.0 * sum_variance))
    p_value = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return float(t_stat), float(p_value)
