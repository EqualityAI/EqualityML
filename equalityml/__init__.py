"""Fairness Assessment and Inequality Reduction tools to help deal with sensitive attributes."""

from .fair import FAIR
from .models_comparison import compare_models

__name__ = "equalityml"
__version__ = '0.1.0a1'
__all__ = ["FAIR"]
