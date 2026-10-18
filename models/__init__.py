# ==============================================================================
# models/__init__.py
# ==============================================================================
"""
Machine Learning Models Module
Supervised and unsupervised learning for bot detection
"""

from .train_supervised import SupervisedDetector
from .train_unsupervised import UnsupervisedDetector
from .predict import BotPredictor

__all__ = ['SupervisedDetector', 'UnsupervisedDetector', 'BotPredictor']