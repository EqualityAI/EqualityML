import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, Union, Dict
from scipy.stats import mstats
from sklearn.utils.multiclass import type_of_target
from collections import defaultdict
import matplotlib.pyplot as plt

# Quantiles for lower bound, curve, and upper bound
QUANTILES_MEDIAN_80 = np.array([0.1, 0.5, 0.9])
DECISION_THRESHOLD = ['f1', 'max']
METRICS = ["f1", "precision", "recall", "queue_rate", "cost"]

class DiscriminationThreshold:
    """
    DiscriminationThreshold computes the threshold  how precision, recall, f1 score, and queue rate change as the
    discrimination threshold increases. For probabilistic, binary classifiers,
    the discrimination threshold is the probability at which you choose the
    positive class over the negative.

    Generally this is set to 50%, but adjusting the discrimination threshold will adjust sensitivity to false
    positives which is described by the inverse relationship of precision and recall with respect to the threshold.
    The visualizer also accounts for variability in the model by running
    multiple trials with different train and test splits of the data. The
    variability is visualized using a band such that the curve is drawn as the
    median score of each trial and the band is from the 10th to 90th
    percentile.
    The visualizer is intended to help users determine an appropriate
    threshold for decision making (e.g. at what threshold do we have a human
    review the data), given a tolerance for precision and recall or limiting
    the number of records to check (the queue rate).
    """

    def __init__(self,
                 model,
                 data,
                 target_variable,
                 decision_threshold=DECISION_THRESHOLD,
                 metrics=["f1"],
                 fair_object=None,
                 utility_costs=None,
                 quantiles=QUANTILES_MEDIAN_80,
                 test_size=0.2,
                 num_thresholds=25,
                 num_iterations=10,
                 random_seed=None):
        """Constructs and checks the values of all the necessary attributes for
        creating a class instance.
        Parameters
        ----------
            model: sklearn.base.BaseEstimator
                Any binary classification model from scikit-learn (or scikit-
                learn pipeline with such a model as the last step) containing
                the method predict_proba() for predicting probability of the
                response.
            data: pd.DataFrame:
                2-dimensional training DataFrame.
            target_variable: str
                Vector with response values.
            test_size: float
                A float value between 0 and 1 corresponding to the share of
                the test set.
            TODO
            random_seed: int
        Returns
        -------
            None
        """
        self.X = data.drop(columns=target_variable)
        self.y = data[target_variable]

        # Check metrics
        self._check_metrics(metrics, utility_costs, fair_object)

        # Check model
        if getattr(model, "_estimator_type", None) != "classifier":
            raise TypeError("Model has to be a classifier")

        # Check data
        if type_of_target(self.y) != "binary":
            raise ValueError("multiclass format is not supported")

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('Prediction and response variables have'
                             'different dimensions.')

        # Check quantiles
        self.quantiles = self._check_quantiles(quantiles)

        # Check decision threshold
        self.decision_threshold = self._check_decision_threhsold(decision_threshold)

        # Check test size
        if not 0 < test_size < 1:
            raise ValueError("The value of the parameter 'test_size' must be "
                             "strictly larger that 0 and smaller than 1")

        # Check number of thresholds
        if not 2 <= num_thresholds <= 1000:
            raise ValueError("The value of the parameter 'num_thresholds' must be "
                             "strictly larger that 2 and smaller than 1000")

        # Check number of iterations
        if not 2 <= num_iterations <= 100:
            raise ValueError("The value of the parameter 'num_iterations' must be "
                             "strictly larger that 2 and smaller than 100")

        # Set params
        self.model = model
        self.test_size = test_size
        self.num_thresholds = num_thresholds
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.discrimination_threshold = None
        self._metrics_quantiles = {}
        self._thresholds = np.linspace(0.0, 1.0, num=self.num_thresholds)

        try:
            self.model.fit(self.X, self.y)
        except ValueError:
            print('Make sure that the model is trained and the input data '
                  'is properly transformed.')

    def _check_metrics(self, metrics, utility_costs, fair_object):
        self.metrics = []
        self.fair_object = None
        self.utility_costs = None
        self.fairness_metric_name = None
        for metric in metrics:
            metric = metric.lower()
            if metric in METRICS:
                if metric == 'cost':
                    if self._check_utility_costs(utility_costs):
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
            else:
                raise f"Invalid metric {metric}"

    def _check_utility_costs(self, utility_costs):
        if utility_costs is not None:
            if len(utility_costs) == 4 and all(isinstance(x, (int, float)) for x in utility_costs):
                return True
        return False

    def _check_quantiles(self, quantiles):
        if len(quantiles) != 3 or not np.all(quantiles[1:] >= quantiles[:-1], axis=0) or not np.all(quantiles < 1):
            raise ValueError("quantiles must be a sequence of three monotonically increasing values less than 1")
        return np.asarray(quantiles)

    def _check_decision_threhsold(self, decision_threshold):
        if decision_threshold and decision_threshold[0] in self.metrics:
            if decision_threshold[1] == 'max' or decision_threshold[1] == 'min':
                return decision_threshold
            elif decision_threshold[1] == 'limit' and decision_threshold[0] in ["queue_rate", "recall", "precision", "f1"]:
                return decision_threshold

        print(f"Invalid decision threshold. Going to use default one: {DECISION_THRESHOLD}")
        return DECISION_THRESHOLD

    def fit(self):
        """
        Fit
        Parameters
        ----------
        Returns
        -------
        """
        rng = np.random.RandomState(self.random_seed)
        metrics = defaultdict(list)
        for _ in tqdm(range(self.num_iterations)):
            randint = rng.randint(low=0, high=32768)
            trial = self._get_metrics(randint)
            for metric, values in trial.items():
                metrics[metric].append(values)

        # Convert metrics to metric arrays
        metrics = {
            metric: np.array(values) for metric, values in metrics.items()
        }

        # Perform aggregation and store cv_scores_
        quantiles = QUANTILES_MEDIAN_80

        for metric, values in metrics.items():

            if metric == 'cost':
                values = (values - np.min(values)) / (np.max(values) - np.min(values))

            # Compute the lower, median, and upper plots
            lower, median, upper = mstats.mquantiles(values, prob=quantiles, axis=0)

            # Store the aggregates in cv scores
            self._metrics_quantiles[metric] = {}
            self._metrics_quantiles[metric]["median"] = median
            self._metrics_quantiles[metric]["lower"] = lower
            self._metrics_quantiles[metric]["upper"] = upper

            # Compute discrimination threshold for metric to maximize
            if self.decision_threshold and self.decision_threshold[0] == metric:
                if self.decision_threshold[1] == 'max':
                    idx = median.argmax()
                    self.discrimination_threshold = self._thresholds[idx]
                elif self.decision_threshold[1] == 'min':
                    idx = median.argmin()
                    self.discrimination_threshold = self._thresholds[idx]
                elif self.decision_threshold[1] == 'limit' and metric in ["queue_rate", "recall"]:
                    idx = next(x[0] for x in enumerate(median) if x[1] <= float(self.decision_threshold[2]))
                    if 0 <= idx < self.num_thresholds:
                        self.discrimination_threshold = self._thresholds[idx]
                elif self.decision_threshold[1] == 'limit' and metric in ["precision", "f1"]:
                    idx = next(x[0] for x in enumerate(median) if x[1] >= float(self.decision_threshold[2]))
                    if 0 <= idx < self.num_thresholds:
                        self.discrimination_threshold = self._thresholds[idx]

        return self.discrimination_threshold

    def _confusion_matrix(self, y_test, pred_label):

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

    def _get_metrics(self, randint):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                            random_state=randint)
        self.model.fit(X_train, y_train)
        predicted_prob = self.model.predict_proba(X_test)[:, 1]
        if self.fair_object:
            self.fair_object.update_classifier(self.model)

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
                tn, fp, fn, tp = self._confusion_matrix(y_test, pred_label)
                pr = tp / (tp + fp) if tp + fp != 0 else 1
                rec = tp / (tp + fn) if tp + fn != 0 else 0
                f1 = 2 / (pr ** (-1) + rec ** (-1)) if pr * rec != 0 else 0
                queue_rate = np.mean(predicted_prob >= threshold)

                precisions.append(pr)
                recalls.append(rec)
                f1_scores.append(f1)
                queue_rates.append(queue_rate)

                if self.utility_costs:
                    array1 = np.array(self.utility_costs)
                    array2 = np.array([tn, fp, fn, tp])
                    cost = np.sum(array1 * array2)
                    costs.append(cost)

            if self.fairness_metric_name:
                self.fair_object.threshold = threshold
                fairness_metric = self.fair_object.fairness_metric(self.fairness_metric_name)
                fairness_metrics.append(fairness_metric)

        if "precision" in self.metrics:
            result["precision"] = precisions
        if "recall" in self.metrics:
            result["recall"] = recalls
        if "f1" in self.metrics:
            result["f1"] = f1_scores
        if "queue_rate" in self.metrics:
            result["queue_rate"] = queue_rates
        if "cost" in self.metrics:
            result['cost'] = costs
        if self.fairness_metric_name:
            result[self.fairness_metric_name] = fairness_metrics

        return result

    def show(self):
        """
        Draws the scores as a line chart on the current axes.
        """
        # Set the colors from the supplied values or reasonable defaults
        cmap = plt.get_cmap("tab10")
        fig = plt.gcf()
        ax = plt.gca()
        for idx, metric in enumerate(self._metrics_quantiles.keys()):

            # Get the color ensuring every metric has a static color
            color = cmap(idx)

            # Make the label pretty
            if metric == "f1":
                label = "$f_1$"
            else:
                label = metric.replace("_", " ")

            # Draw the metric values
            ax.plot(
                self._thresholds, self._metrics_quantiles[metric]["median"], color=color, label=label
            )

            # Draw the upper and lower bounds
            ax.fill_between(
                self._thresholds, self._metrics_quantiles[metric]["upper"], self._metrics_quantiles[metric]["lower"],
                alpha=0.35, linewidth=0, color=color
            )

            # Annotate the graph with the maximizing value
            if self.discrimination_threshold and self.decision_threshold[0] == metric:
                ax.axvline(
                    self.discrimination_threshold,
                    ls="--",
                    c="k",
                    lw=1,
                    label="$t_{}={:0.2f}$".format(metric[0], self.discrimination_threshold),
                )
        # Set the title of the threshold visualization
        ax.set_title("Threshold Plot for {}".format(self.model.__class__.__name__))

        ax.legend(frameon=True, loc="best")
        ax.set_xlabel("discrimination threshold")
        ax.set_ylabel("score")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        savefig = False
        if savefig:
            plt.savefig()
        else:
            plt.show()

        clear_figure = False
        if clear_figure:
            fig.clear()


def discrimination_threshold(
        model,
        data,
        target_variable,
        decision_threshold=('f1', 'max'),
        metrics=['f1'],
        fair_object=None,
        utility_costs=None,
        show=False,
        test_size=0.2,
        num_thresholds=25,
        num_iterations=10,
        random_seed=None
):
    """Discrimination Threshold
    TODO
    Visualizes how precision, recall, f1 score, and queue rate change as the
    discrimination threshold increases. For probabilistic, binary classifiers,
    the discrimination threshold is the probability at which you choose the
    positive class over the negative. Generally this is set to 50%, but
    adjusting the discrimination threshold will adjust sensitivity to false
    positives which is described by the inverse relationship of precision and
    recall with respect to the threshold.
    Parameters
    ----------
    num_thresholds
    model : estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values. The target y must
        be a binary classification target.
    num_iterations : integer, default: 50
        Number of times to shuffle and split the dataset to account for noise
        in the threshold metrics curves. Note if cv provides > 1 splits,
        the number of trials will be n_trials * cv.get_n_splits()
    argmax : str or None, default: 'fscore'
        Annotate the threshold maximized by the supplied metric (see exclude
        for the possible metrics to use). If None or passed to exclude,
        will not annotate the graph.
    quantiles : sequence, default: np.array([0.1, 0.5, 0.9])
        Specify the quantiles to view model variability across a number of
        trials. Must be monotonic and have three elements such that the first
        element is the lower bound, the second is the drawn curve, and the
        third is the upper bound. By default the curve is drawn at the median,
        and the bounds from the 10th percentile to the 90th percentile.
    random_seed : int, optional
        Used to seed the random state for shuffling the data while composing
        different train and test splits. If supplied, the random state is
        incremented in a deterministic fashion for each split.
        Note that if a splitter is provided, it's random state will also be
        updated with this random state, even if it was previously set.
    show : bool, default: True
        If True, calls ``show()``, which in turn calls ``plt.show()`` however you cannot
        call ``plt.savefig`` from this signature, nor ``clear_figure``. If False, simply
        calls ``finalize()``
    Returns
    -------
    threshold : float
        Returns the discrimination threshold based on decision threshold input
    """

    # Instantiate the DiscriminationThreshold
    dt = DiscriminationThreshold(model,
                                 data,
                                 target_variable,
                                 decision_threshold=decision_threshold,
                                 metrics=metrics,
                                 fair_object=fair_object,
                                 utility_costs=utility_costs,
                                 test_size=test_size,
                                 num_thresholds=num_thresholds,
                                 num_iterations=num_iterations,
                                 random_seed=random_seed)

    # Fit the DiscriminationThreshold
    threshold = dt.fit()

    if show:
        dt.show()

    # Return the discrimination threshold value
    return threshold


def binary_threshold_score(scoring, ml_model, X, y, threshold=0.5):
    """
    Binary classification score.
    Computes the score of the binary classification based on input discrimination threshold.
    Parameters
    ----------
    scoring : str, callable. Default=None
        If None (default), uses 'accuracy' for sklearn classifiers
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc}.
        If a callable object or function is provided, it has to agree with
        sklearn's signature 'scorer(estimator, X, y)'. Check
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.
    ml_model : a Scikit-Learn estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised.
    X : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features
    y : ndarray or Series of length n
        An array or series of target or class values. The target y must
        be a binary classification target.
    threshold : float, default=0.5
        Discrimination threshold for predicting the favorable class.
    Returns
    ----------
    score : float
        Binary classification score
    """
    if ml_model._estimator_type != "classifier":
        raise AttributeError("Model must be a Classifier.")

    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    _pred_class = np.asarray(list(map(lambda x: 1 if x > threshold else 0, ml_model.predict_proba(X)[:, -1])))
    score = scorer._score_func(y, _pred_class)
    return score
