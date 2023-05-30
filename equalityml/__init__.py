"""Fairness Assessment and Inequality Reduction tools to help deal with sensitive attributes."""

from .fair import FAIR
from .stats import paired_ttest
from .threshold import discrimination_threshold, binary_threshold_score


__name__ = "equalityml"
__version__ = '0.2.0'
__all__ = ["FAIR"]
