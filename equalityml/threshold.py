import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from scipy.stats import mstats
from collections import defaultdict
import matplotlib.pyplot as plt

from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

# Quantiles for lower bound, curve, and upper bound
QUANTILES_MEDIAN_80 = np.array([0.1, 0.5, 0.9])
# Default decision maker
DECISION_MAKER = ('f1', 'max')
# Available metrics
METRICS = ("accuracy", "f1", "precision", "recall", "queue_rate", "cost")


# Utilities functions
def _check_utility_costs(utility_costs):
    if utility_costs is not None:
        if len(utility_costs) == 4 and all(isinstance(x, (int, float)) for x in utility_costs):
            return utility_costs
    raise ValueError(f"Invalid utility costs {utility_costs}. It must be a list of 4 values for [TP FN FP TN]")


def _check_quantiles(quantiles):
    if len(quantiles) != 3 or not np.all(quantiles[1:] >= quantiles[:-1], axis=0) or not np.all(quantiles < 1):
        raise ValueError("Quantiles must be a sequence of three monotonically increasing values less than 1")
    return np.asarray(quantiles)


def _confusion_matrix(y_test, pred_label):
    """
    Compute confusion matrix to evaluate the accuracy of a binary classification.
    Returns TN, FP, FN, TP
    """
    y_test = list(y_test)
    pred_label = list(pred_label)
    if len(set(y_test)) > 2 or len(set(pred_label)) > 2:
        raise AttributeError("Inputs should be binary.")

    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(len(pred_label)):  # the confusion matrix is for 2 classes
        # 1=positive, 0=negative
        if int(pred_label[i]) == 1 and int(y_test[i]) == 1:
            tp += 1  # True Positives
        elif int(pred_label[i]) == 1 and int(y_test[i]) == 0:
            fp += 1  # False Positives
        elif int(pred_label[i]) == 0 and int(y_test[i]) == 1:
            fn += 1  # False Negatives
        elif int(pred_label[i]) == 0 and int(y_test[i]) == 0:
            tn += 1  # True Negatives

    return tn, fp, fn, tp


class DiscriminationThreshold:
    """
    The DiscriminationThreshold class provides a solution for determining the optimal discrimination threshold in a
    binary classification model for decision makers. The discrimination threshold refers to the probability value that
    separates the positive and negative classes. The commonly used threshold is 0.5, however, adjusting it will affect
    the sensitivity to false positives, as precision and recall exhibit an inverse relationship with respect to the
    threshold. This class facilitates the selection of the appropriate threshold for decision-making purposes, such as
    determining the threshold at which the human has to review the data or maximizing the f1 score.

    Notes
    ----------
    The optimal discrimination threshold also accounts for variability in the model by running multiple trials with
    different train and test splits of the data. The variability can be visualized using a band such that the curve is
    drawn as the median score of each trial and the band is from the 10th to 90th percentile.

    Parameters
    ----------
    ml_model : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values. The target y must be a binary classification target.
    decision_maker : tuple, default=('f1', 'max')
        The metric and decision to optimize the discrimination threshold. The metric shall be available in input metrics
        list and 3 different decisions can be used: 'max', 'min' or 'limit' with the following behaviour:
        - 'max' computes the threshold which maximizes the selected metric
        - 'min' computes the threshold which minimizes the selected metric
        - 'limit' requires an extra float parameter between 0 and 1. The optimal threshold is calculated when the
        selected metric reaches that limit.
    metrics : tuple, default='f1'
        List of metrics to evaluate the model. Available options are: "accuracy", "f1", "precision", "recall",
        "queue_rate", "cost" (which requires the utility_costs parameter) and fairness metrics (which requires a FAIR
        object parameter)
    fair_object : object, default=None
        FAIR object to calculate the fairness metric. It is only used when a fairness metric is provided in metrics
        input parameter.
    utility_costs : list, default=None
        Utility costs for cost-sensitive learning. It has to be a 4 element list where the cost values correspond to the
        following cost sequence: [TP, FN, FP, TN]
    quantiles : sequence, default=np.array([0.1, 0.5, 0.9])
        Specify the quantiles to view model variability across a number of trials. Must be monotonic and have three
        elements such that the first element is the lower bound, the second is the drawn curve, and the third is the
        upper bound. By default the curve is drawn at the median, and the bounds from the 10th percentile to the 90th
        percentile.
    test_size : float, default=0.2
        Proportion of data to be used for testing. The data split is performed using the 'train_test_split' function
        from sklearn package, in a stratified fashion.
    num_thresholds : int, default=100
        Number of thresholds to consider which are evenly spaced over the interval [min_bound, max_bound].
    min_bound: float, default=0.0
        Minimum threshold bound
    max_bound: float, default=1.0
        Minimum threshold bound
    num_iterations : int, default=10
        Number of times to shuffle and split the dataset to account for noise in the threshold metrics curves.
        If training model is not required, the model will be evaluated once.
    model_training : bool, default=True
        When True, the model is trained 'num_iterations' times to get the metrics variability, otherwise, the model will
        be only evaluated once.
    random_seed : int, default=None
        Used to seed the random state for splitting the data in different train and test splits. If supplied, the
        random state is incremented in a deterministic fashion for each split.

    """

    def __init__(self,
                 ml_model,
                 X,
                 y,
                 decision_maker=DECISION_MAKER,
                 metrics=("f1"),
                 fair_object=None,
                 utility_costs=None,
                 quantiles=QUANTILES_MEDIAN_80,
                 test_size=0.2,
                 num_thresholds=100,
                 min_bound=0.0,
                 max_bound=1.0,
                 num_iterations=10,
                 model_training=True,
                 random_seed=None):

        super(DiscriminationThreshold, self).__init__()

        self.model_training = model_training
        if self.model_training is False:
            num_iterations = 1
            if not hasattr(ml_model, "classes_"):
                raise ValueError("When model will not be trained, it must be fitted.")

        self.X = X.copy()
        self.y = y.copy()

        # Check metrics
        self._check_metrics(metrics, utility_costs, fair_object)

        # Check model
        if getattr(ml_model, "_estimator_type", None) != "classifier":
            raise TypeError("Model has to be a classifier")

        # Check data
        if type_of_target(self.y) != "binary":
            raise ValueError("multiclass format is not supported")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('Prediction and response variables have'
                             'different dimensions.')

        # Check quantiles
        self.quantiles = _check_quantiles(quantiles)

        # Check decision maker
        self.decision_maker = self._check_decision_maker(decision_maker)

        # Check test size
        if not 0 < test_size < 1:
            raise ValueError("The value of the parameter 'test_size' must be "
                             "strictly larger that 0 and smaller than 1")

        # Check number of thresholds
        if not 2 <= num_thresholds <= 1000:
            raise ValueError("The value of the parameter 'num_thresholds' must be "
                             "strictly larger that 2 and smaller than 1000")

        if not 0.0 <= min_bound < max_bound <= 1.0:
            raise ValueError("Invalid min_bound/max_bound value. The condition 0.0 <= min_bound < max_bound <= 1.0 "
                             "must comply.")

        # Check number of iterations
        if not 1 <= num_iterations <= 100:
            raise ValueError("The value of the parameter 'num_iterations' must be "
                             "strictly larger that 1 and smaller than 100")

        # Set params
        self.ml_model = copy.deepcopy(ml_model)
        self.test_size = test_size
        self.num_thresholds = num_thresholds
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.discrimination_threshold = None
        self._metrics_quantiles = defaultdict(dict)
        self._thresholds = np.linspace(min_bound, max_bound, num=self.num_thresholds)

        if self.model_training:
            try:
                self.ml_model.fit(self.X, self.y)
            except ValueError:
                print('Make sure that the model is trained and the input data '
                      'is properly transformed.')

    def _check_metrics(self,
                       metrics,
                       utility_costs,
                       fair_object):
        """
        Verify input metrics which shall exist in METRICS or be a fairness metric.
        If the metric 'cost' is passed, a valid utility cost is required.
        If a fairness metric is passed, a FAIR object is required.
        """
        self.metrics = []
        self.fair_object = None
        self.utility_costs = None
        self.fairness_metric_name = None
        for metric in metrics:
            metric = metric.lower()
            if metric in METRICS:
                if metric == 'cost':
                    if _check_utility_costs(utility_costs):
                        self.metrics.append(metric)
                        self.utility_costs = utility_costs
                    else:
                        raise f"Invalid utility costs {utility_costs}"
                else:
                    self.metrics.append(metric)
            elif fair_object is not None and metric in fair_object.fairness_metrics_list:
                self.metrics.append(metric)
                self.fair_object = fair_object
                self.fairness_metric_name = metric

        if len(self.metrics) == 0:
            raise f"Invalid input metrics {metrics}"

    def _check_decision_maker(self,
                              decision_maker):
        if decision_maker and decision_maker[0] in self.metrics and len(decision_maker) >= 2:
            if decision_maker[1] in ['max', 'min']:
                return decision_maker
            elif decision_maker[1] == 'limit':
                if len(decision_maker) == 3 and 0 < float(decision_maker[2]) < 1:
                    return decision_maker

        print(f"Invalid decision maker. Going to use default one: {DECISION_MAKER}")
        if 'f1' not in self.metrics:
            self.metrics.append('f1')
        return DECISION_MAKER

    def _get_metrics(self,
                     randint):
        """
        Helper function for the internal method fit() which performs row-wise calculation of accuracy, precision,
        recall, F1 score and queue rate, utility cost and fairness metric.
        """
        if self.model_training:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                                random_state=randint, stratify=self.y)
            self.ml_model.fit(X_train, y_train)
        else:
            X_test, y_test = self.X, self.y
        predicted_prob = self.ml_model.predict_proba(X_test)[:, 1]
        if self.fair_object:
            self.fair_object.update_classifier(self.ml_model)

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        queue_rates = []
        costs = []
        if self.fairness_metric_name:
            fairness_metrics = []

        result = {}
        for threshold in self._thresholds:
            if any(metric in self.metrics for metric in METRICS):
                pred_label = [1 if (prob >= threshold) else 0 for prob in predicted_prob]
                tn, fp, fn, tp = _confusion_matrix(y_test, pred_label)
                acc = (tp + tn) / (tn + fp + fn + tp)
                pr = tp / (tp + fp) if tp + fp != 0 else 1
                rec = tp / (tp + fn) if tp + fn != 0 else 0
                f1 = 2 / (pr ** (-1) + rec ** (-1)) if pr * rec != 0 else 0
                queue_rate = np.mean(predicted_prob >= threshold)

                accuracies.append(acc)
                precisions.append(pr)
                recalls.append(rec)
                f1_scores.append(f1)
                queue_rates.append(queue_rate)

                if self.utility_costs:
                    array1 = np.array(self.utility_costs)
                    array2 = np.array([tp, fn, fp, tn])
                    cost = np.sum(array1 * array2)
                    costs.append(cost)

            if self.fairness_metric_name:
                self.fair_object.threshold = threshold
                fairness_metric = self.fair_object.fairness_metric(self.fairness_metric_name)
                fairness_metrics.append(fairness_metric)

        if "accuracy" in self.metrics:
            result["accuracy"] = accuracies
        if "precision" in self.metrics:
            result["precision"] = precisions
        if "recall" in self.metrics:
            result["recall"] = recalls
        if "f1" in self.metrics:
            result["f1"] = f1_scores
        if "queue_rate" in self.metrics:
            result["queue_rate"] = queue_rates
        if "cost" in self.metrics:
            if len(self.metrics) == 1:
                result['cost'] = costs
            else:
                result['cost'] = [(float(cost)-min(costs))/(max(costs)-min(costs)) for cost in costs]
        if self.fairness_metric_name:
            result[self.fairness_metric_name] = fairness_metrics

        return result

    def fit(self):
        """
        Fit method computes the optimal discrimination threshold by aggregating the metrics calculated at all
        'num_iterations' trials.
        For each trial, the dataset is shuffled and split and all required metrics like: accuracy, precision, recall,
        f1, queue rate, utility cost, and fairness metric scores are calculated. The scores are aggregated by the
        quantiles expressed. Finally, the discrimination threshold value is computed based on provided decision
        threshold rule.
        """
        rng = np.random.RandomState(self.random_seed)
        metrics = defaultdict(list)

        # Iterate over num_iterations and compute the metrics for each iteration
        if self.num_iterations > 1:
            for _ in tqdm(range(self.num_iterations)):
                randint = rng.randint(low=0, high=32768)
                trial = self._get_metrics(randint)
                for metric, values in trial.items():
                    metrics[metric].append(values)
        else:
            randint = rng.randint(low=0, high=32768)
            trial = self._get_metrics(randint)
            for metric, values in trial.items():
                metrics[metric].append(values)

        # Convert metrics to metric arrays
        metrics = {
            metric: np.array(values) for metric, values in metrics.items()
        }

        # Perform aggregation and store _metrics_quantiles
        for metric, values in metrics.items():

            # Compute the lower, median, and upper plots
            lower, median, upper = mstats.mquantiles(values, prob=self.quantiles, axis=0)

            # Store the aggregates in _metrics_quantiles
            self._metrics_quantiles[metric] = {}
            self._metrics_quantiles[metric]["median"] = median
            self._metrics_quantiles[metric]["lower"] = lower
            self._metrics_quantiles[metric]["upper"] = upper

            # Compute discrimination threshold
            if self.decision_maker and self.decision_maker[0] == metric:
                if self.decision_maker[1] == 'max':
                    idx = median.argmax()
                    self.discrimination_threshold = [self._thresholds[idx], median.max()]
                elif self.decision_maker[1] == 'min':
                    idx = median.argmin()
                    self.discrimination_threshold = [self._thresholds[idx], median.min()]
                elif self.decision_maker[1] == 'limit' and metric in ["queue_rate", "recall", "costs"]:
                    x = next(x for x in enumerate(median) if x[1] <= float(self.decision_maker[2]))
                    self.discrimination_threshold = [self._thresholds[x[0]], x[1]]
                elif self.decision_maker[1] == 'limit' and metric in ["precision", "f1"]:
                    x = next(x for x in enumerate(median) if x[1] >= float(self.decision_maker[2]))
                    self.discrimination_threshold = [self._thresholds[x[0]], x[1]]

        return self.discrimination_threshold[0]

    def show(self):
        """
        Draws the metrics scores as a line chart and annotate the graph with the discrimination threshold value.
        """
        # Set the colors from the supplied values or reasonable defaults
        cmap = plt.get_cmap("tab10")
        fig = plt.gcf()
        ax = plt.gca()
        for idx, metric in enumerate(self._metrics_quantiles.keys()):

            # Get the color ensuring every metric has a static color
            color = cmap(idx)

            # Make the label pretty
            label = metric.replace("_", "-")

            # Draw the metric values
            if self.decision_maker[0] == metric:
                label = "${}={:0.3f}$".format(label, self.discrimination_threshold[1])
            ax.plot(
                self._thresholds, self._metrics_quantiles[metric]["median"], color=color, label=label
            )

            # Draw the upper and lower bounds
            ax.fill_between(
                self._thresholds, self._metrics_quantiles[metric]["upper"], self._metrics_quantiles[metric]["lower"],
                alpha=0.35, linewidth=0, color=color
            )

            # Annotate the graph with the discrimination threshold value
            if self.discrimination_threshold and self.decision_maker[0] == metric:
                ax.axvline(
                    self.discrimination_threshold[0],
                    ls="--",
                    c="k",
                    lw=1,
                    label="$t_{}={:0.3f}$".format(metric[0], self.discrimination_threshold[0]),
                )
        # Set the title of the threshold visualization
        ax.set_title("Threshold Plot for {}".format(self.ml_model.__class__.__name__))

        ax.legend(frameon=True, loc="best")
        ax.set_xlabel("discrimination threshold")
        ax.set_ylabel("score")
        ax.set_xlim(0.0, 1.0)
        #ax.set_ylim(0.0, 1.0)

        savefig = False
        if savefig:
            plt.savefig()
        else:
            plt.show()

        clear_figure = False
        if clear_figure:
            fig.clear()


def discrimination_threshold(
        ml_model,
        X,
        y,
        decision_maker=DECISION_MAKER,
        metrics=("f1"),
        fair_object=None,
        utility_costs=None,
        quantiles=QUANTILES_MEDIAN_80,
        test_size=0.2,
        num_thresholds=100,
        min_bound=0.0,
        max_bound=1.0,
        num_iterations=10,
        model_training=True,
        random_seed=None,
        show=False
):
    """
    The discrimination_threshold function provides a solution for determining the optimal discrimination threshold in a
    binary classification model for decision makers. The discrimination threshold refers to the probability value that
    separates the positive and negative classes. The commonly used threshold is 0.5, however, adjusting it will affect
    the sensitivity to false positives, as precision and recall exhibit an inverse relationship with respect to the
    threshold. This function facilitates the selection of the appropriate threshold for decision-making purposes, such
    as determining the threshold at which the human has to review the data or maximizing the f1 score.
    See DiscriminationThreshold class for more details.

    Parameters
    ----------
    ml_model : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values. The target y must be a binary classification target.
    decision_maker : tuple, default=('f1', 'max')
        The metric and decision to optimize the discrimination threshold. The metric shall be available in input metrics
        list and 3 different decisions can be used: 'max', 'min' or 'limit' with the following behaviour:
        - 'max' computes the threshold which maximizes the selected metric
        - 'min' computes the threshold which minimizes the selected metric
        - 'limit' requires an extra float parameter between 0 and 1. The optimal threshold is calculated when the
        selected metric reaches that limit.
    metrics : tuple, default='f1'
        List of metrics to evaluate the model. Available options are: "accuracy", "f1", "precision", "recall",
        "queue_rate", "cost" (which requires the utility_costs parameter) and fairness metrics (which requires a FAIR
        object parameter)
    fair_object : object, default=None
        FAIR object to calculate the fairness metric. It is only used when a fairness metric is provided in metrics
        input parameter.
    utility_costs : list, default=None
        Utility costs for cost-sensitive learning. It has to be a 4 element list where the cost values correspond to the
        following cost sequence: [TP, FN, FP, TN]
    quantiles : sequence, default=np.array([0.1, 0.5, 0.9])
        Specify the quantiles to view model variability across a number of trials. Must be monotonic and have three
        elements such that the first element is the lower bound, the second is the drawn curve, and the third is the
        upper bound. By default the curve is drawn at the median, and the bounds from the 10th percentile to the 90th
        percentile.
    test_size : float, default=0.2
        Proportion of data to be used for testing. The data split is performed using the 'train_test_split' function
        from sklearn package, in a stratified fashion.
    num_thresholds : int, default=100
        Number of thresholds to consider which are evenly spaced over the interval [min_bound, max_bound].
    min_bound: float, default=0.0
        Minimum threshold bound
    max_bound: float, default=1.0
        Minimum threshold bound
    num_iterations : int, default=10
        Number of times to shuffle and split the dataset to account for noise in the threshold metrics curves.
        If training model is not required, the model will be evaluated once.
    model_training : bool, default=True
        When True, the model is trained 'num_iterations' times to get the metrics variability, otherwise, the model will
        be only evaluated once.
    random_seed : int, default=None
        Used to seed the random state for splitting the data in different train and test splits. If supplied, the
        random state is incremented in a deterministic fashion for each split.
    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()``.
    Returns
    -------
    threshold : float
        Returns the discrimination threshold complying with the provided decision maker.
    """

    # Instantiate the DiscriminationThreshold
    dt = DiscriminationThreshold(ml_model,
                                 X,
                                 y,
                                 decision_maker=decision_maker,
                                 metrics=metrics,
                                 fair_object=fair_object,
                                 utility_costs=utility_costs,
                                 quantiles=quantiles,
                                 test_size=test_size,
                                 num_thresholds=num_thresholds,
                                 min_bound=min_bound,
                                 max_bound=max_bound,
                                 num_iterations=num_iterations,
                                 model_training=model_training,
                                 random_seed=random_seed)

    # Fit the DiscriminationThreshold
    threshold = dt.fit()

    if show:
        dt.show()

    # Return the discrimination threshold value
    return threshold


def binary_threshold_score(ml_model,
                           X,
                           y,
                           scoring=None,
                           threshold=0.5,
                           utility_costs=None):
    """
    Binary classification score.
    Computes the score of the binary classification based on input discrimination threshold.
    Parameters
    ----------
    ml_model : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is not a classifier, an exception is raised.
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values. The target y must be a binary classification target.
    scoring : str, callable. Default=None
        If None (default), uses 'accuracy' for sklearn classifiers
        If 'cost', uses utility_costs parameter to calculate the score
        If str, uses a sklearn scoring metric string identifier, for example {accuracy, f1, precision, recall, roc_auc}.
        If a callable object or function is provided, it has to agree with sklearn's signature 'scorer(estimator, X, y)'.
        Check http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html for more information.
    threshold : float, default=0.5
        Discrimination threshold at which the model predicts the target as positive over the negative.
    utility_costs : list, default=None
        Utility costs for cost-sensitive learning. It has to be a 4 element list where the cost values correspond to the
        following cost sequence: [TP, FN, FP, TN]
    Returns
    ----------
    score : float
        Binary classification score
    """
    if ml_model._estimator_type != "classifier":
        raise AttributeError("Model must be a Classifier.")

    # Get scoring when is None
    if scoring is None:
        if ml_model._estimator_type == "classifier":
            scoring = "accuracy"
        elif ml_model._estimator_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Model must be a Classifier or Regressor.")

    _pred_class = np.asarray(list(map(lambda x: 1 if x > threshold else 0, ml_model.predict_proba(X)[:, -1])))
    if scoring.lower() == 'cost':
        utility_costs = _check_utility_costs(utility_costs)
        tn, fp, fn, tp = _confusion_matrix(y, _pred_class)
        score = np.sum(np.array(utility_costs) * np.array([tp, fn, fp, tn]))
    else:
        if isinstance(scoring, str):
            scorer = get_scorer(scoring)
        else:
            scorer = scoring

        score = scorer._score_func(y, _pred_class)
    return score
