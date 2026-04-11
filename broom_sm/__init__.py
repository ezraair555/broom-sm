"""
broom-sm: Tidy Statistical Modeling and Inference for Python.
Bridging the gap between R's infer, DABEST estimation, and Pingouin analytics.
"""

__version__ = "0.1.0"

from .infer import specify, hypothesize, generate, calculate
from .estimation import load_dabest, estimation_plot
from .reporting import tidy, glance, augment
