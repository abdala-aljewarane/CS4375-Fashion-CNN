# __init__.py for the main code package

"""
CNN Architecture Comparison for Fashion Classification
CS-4375 Course Project

This package contains code for training and evaluating
CNN models on fashion datasets.
"""

__version__ = '0.1.0'
__author__ = 'Abdala Aljewarane, Lerich Osay'

# Import key components for easier access
from .train import train_model
from .evaluate import evaluate_model