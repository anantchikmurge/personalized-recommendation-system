"""
Personalized Recommendation System
A comprehensive machine learning-based recommendation engine
"""

__version__ = "1.0.0"
__author__ = "Recommendation System Team"
__description__ = "A comprehensive recommendation system with multiple algorithms"

from . import data_processing
from . import models
from . import evaluation
from . import dashboard

__all__ = [
    'data_processing',
    'models', 
    'evaluation',
    'dashboard'
] 