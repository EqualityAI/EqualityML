from sklearn.metrics import get_scorer
from yellowbrick.classifier.threshold import discrimination_threshold as dt
import numpy as np
import matplotlib.pyplot as plt


## TODO Add discrimination threshold computation

def discrimination_threshold(model, data, target_variable, metric="score", scoring=None, fair_object=None, metric_name=None):
    X = data.drop(columns=target_variable)
    y = data[target_variable]

    if metric == "score":
        #1- For infinite resource cases: Threshold based on metrics we find interesting (F1, etc)
        if scoring is None:
            if model._estimator_type == "classifier":
                scoring = "accuracy"
            else:
                raise AttributeError("Model must be a Classifier.")

        dt(model, X, y, )
    elif metric == "queue_rate":
        # 2- For rationed resources: Top N ("Queue rate")
        dt(model, X, y)
    elif metric == "cost":
        # 3- For business use cases (with expenses): Utility (cost) optimization
        dt(model, X, y)
    elif metric == "fairness_metric":
        if fair_object is None:
            raise "FAIR object can not be None"
        if metric_name not in fair_object.fairness_metrics_list:
            raise f"Invalid metric name {metric_name}. It's not available in FAIR class"

        def get_metrics(threshold):
            fair_object.set_threshold(threshold)
            fairness_metric = fair_object.fairness_metric(metric_name)

            # # Compute metrics
            # predicted_prob = model.predict_proba(X)[:, 1]
            # pred_label = (predicted_prob >= threshold).astype(int)
            # tp = np.sum((pred_label == 1) & (y == 1))
            # fp = np.sum((pred_label == 1) & (y == 0))
            # fn = np.sum((pred_label == 0) & (y == 1))
            # pr = tp / (tp + fp) if tp + fp != 0 else 1
            # rec = tp / (tp + fn) if tp + fn != 0 else 0
            # f1 = 2 / (pr ** (-1) + rec ** (-1)) if pr * rec != 0 else 0
            # queue_rate = np.mean(predicted_prob >= threshold)

            return fairness_metric

        thresholds = np.arange(0, 1.01, 0.02)
        metrics = list(map(get_metrics, thresholds))

        plot = True
        if plot:
            plt.scatter(thresholds, metrics)
            title = f"Fairness metric '{metric_name}' vs discrimination threshold"
            plt.title(title)
            plt.xlabel("discrimination threshold")
            plt.ylabel(metric_name)
            plt.show()


"""The module contains the class InteractiveDiscriminationThreshold
for constructing Plotly dashboard with an interactive DT plot.
"""

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from typing import Tuple, Union
from scipy.stats import mstats
from sklearn.utils.multiclass import type_of_target
from collections import defaultdict


# Quantiles for lower bound, curve, and upper bound
QUANTILES_MEDIAN_80 = np.array([0.1, 0.5, 0.9])


class DiscriminationThreshold:
    """A class to build a discrimination threshold.
    """

    def __init__(self,
                 model: ClassifierMixin,
                 data: pd.DataFrame,
                 target_variable: str,
                 fair_object: None,
                 fairness_metric_name: "",
                 test_size: float = 0.2,
                 random_seed: int = None):
        """Constructs and checks the values of all the necessary attributes for
        creating a class instance.
        Parameters
        ----------
            model: sklearn.base.ClassifierMixin
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
            random_seed: int
        Returns
        -------
            None
        """
        self.model = model
        self.X = data.drop(columns=target_variable)
        self.y = data[target_variable]
        self.random_seed = random_seed
        self.test_size = test_size
        self.num_thresholds = 50
        self.n_iterations = 2

        self._thresholds = np.linspace(0.0, 1.0, num=self.num_thresholds)
        self._max_metric = 'fscore'
        self.fbeta = 1.0
        self.discrimination_threshold = None
        self._metrics_quantiles = {}
        self.metrics_name = ["precision", "recall", "fscore", "queue_rate"]
        self.fair_object = fair_object
        self.fairness_metric_name = fairness_metric_name

        if fair_object is None:
            self.fairness_metric = False
        else:
            if fairness_metric_name not in fair_object.fairness_metrics_list:
                raise f"Invalid metric name {fairness_metric_name}. It's not available in FAIR class"
        self.fairness_metric = True

        # Check input arguments
        # Check target before metrics raise crazy exceptions
        if getattr(model, "_estimator_type", None) != "classifier":
            raise TypeError("Model has to be a classifier")
        if type_of_target(self.y) != "binary":
            raise ValueError("multiclass format is not supported")

        # Check the various inputs
        #self._check_quantiles(quantiles)
        #self._check_cv(cv)
        #self._check_exclude(exclude)
        #self._check_argmax(argmax, exclude)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError('Prediction and response variables have'
                             'different dimensions.')
        if not 0 < self.test_size < 1:
            raise ValueError("The value of the parameter test_size must be "
                             "strictly larger that 0 and smaller than 1")
        try:
            self.model.fit(self.X, self.y)
        except ValueError:
            print('Make sure that the model is trained and the input data '
                  'is properly transformed.')

    def fit(self,):
        """
        Fit is the entry point for the visualizer. Given instances described
        by X and binary classes described in the target y, fit performs n
        trials by shuffling and splitting the dataset then computing the
        precision, recall, f1, and queue rate scores for each trial. The
        scores are aggregated by the quantiles expressed then drawn.
        Parameters
        ----------
        X : ndarray or DataFrame of shape n x m
            A matrix of n instances with m features
        y : ndarray or Series of length n
            An array or series of target or class values. The target y must
            be a binary classification target.
        kwargs: dict
            keyword arguments passed to Scikit-Learn API.
        Returns
        -------
        self : instance
            Returns the instance of the visualizer
        raises: YellowbrickValueError
            If the target y is not a binary classification target.
        """
        rng = np.random.RandomState(self.random_seed)
        metrics = defaultdict(list)
        for iter in tqdm(range(self.n_iterations)):
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
            # Compute the lower, median, and upper plots
            lower, median, upper = mstats.mquantiles(values, prob=quantiles, axis=0)

            # Store the aggregates in cv scores
            self._metrics_quantiles[metric] = {}
            self._metrics_quantiles[metric]["median"] = median
            self._metrics_quantiles[metric]["lower"] = lower
            self._metrics_quantiles[metric]["upper"] = upper

            # Compute discrimination threshold for metric to maximize
            if self._max_metric and self._max_metric == metric:
                argmax = median.argmax()
                self.discrimination_threshold = self._thresholds[argmax]

        draw_plot = True
        if draw_plot:
            # Draw
            self.draw()

        return self.discrimination_threshold

    def _get_metrics(self, randint: int) -> Tuple:
        """Helper function
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size,
                                                            random_state=randint)
        self.model.fit(X_train, y_train)
        predicted_prob = self.model.predict_proba(X_test)[:, 1]
        self.fair_object.update_classifier(self.model)

        precisions = []
        recalls = []
        f_scores = []
        queue_rates = []
        if self.fairness_metric:
            fairness_metrics = []

        for threshold in self._thresholds:
            pred_label = [1 if (prob >= threshold) else 0 for prob in predicted_prob]
            CM = confusion_matrix(y_test, pred_label)
            fn = CM[1][0]
            tp = CM[1][1]
            fp = CM[0][1]
            pr = tp / (tp + fp) if tp + fp != 0 else 1
            rec = tp / (tp + fn) if tp + fn != 0 else 0
            f1 = 2 / (pr ** (-1) + rec ** (-1)) if pr * rec != 0 else 0
            queue_rate = np.mean(predicted_prob >= threshold)

            precisions.append(pr)
            recalls.append(rec)
            f_scores.append(f1)
            queue_rates.append(queue_rate)

            if self.fairness_metric:
                self.fair_object.set_threshold(threshold)
                fairness_metric = self.fair_object.fairness_metric(self.fairness_metric_name)
                fairness_metrics.append(fairness_metric)

        result = {
            "precision": precisions,
            "recall": recalls,
            "fscore": f_scores,
            "queue_rate": queue_rates,
        }
        if self.fairness_metric:
            result[self.fairness_metric_name] = fairness_metrics

        return result

    def draw(self):
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
            if metric == "fscore":
                if self.fbeta == 1.0:
                    label = "$f_1$"
                else:
                    label = "$f_{{\beta={:0.1f}}}".format(self.fbeta)
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
            if self._max_metric and self._max_metric == metric:
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

