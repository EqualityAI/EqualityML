"""Fairness Assessment and Inequality Reduction tools to help deal with sensitive attributes."""

from .fair import FAIR
from .stats import paired_ttest
from .threshold import DiscriminationThreshold, discrimination_threshold


__name__ = "equalityml"
__version__ = '0.1.0a1'
__all__ = ["FAIR"]
