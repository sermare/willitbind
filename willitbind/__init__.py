"""
WillItBind: Predicting experimental protein binding from computational features.

A toolkit for analyzing 5,000+ de novo protein binder designs with experimental
validation, answering the question every protein engineer asks: will it bind?
"""

__version__ = "1.0.0"
__author__ = "Sergio Martinez"

from .data import BinderDataset
from .features import FeatureAnalyzer
from .models import BindingPredictor, GreedySelector
from .plots import WillItPlot
