"""Bias Mitigation and Fairness evaluation tools to help deal with sensitive attributes."""

from .bias_mitigation import BiasMitigation
from .fairness_evaluation import FairnessMetric

__name__ = "equalityml"
__version__ = '0.1.0'
__all__ = ["FairnessMetric", "BiasMitigation"]
