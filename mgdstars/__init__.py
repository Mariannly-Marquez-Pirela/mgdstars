"""
MGDStars: A Python package for Minimal Geometrical Deformation star calculations.
"""

__version__ = "0.1.0"
__author__ = "Mariannly Marquez"
__email__ = "mariannly.marquez@yachaytech.edu.ec"

# Import main classes for easy access
from .core import MGDStars
from .visualization import StarPlotter, AnimationMaker

# Define what gets imported with "from mgdstars import *"
__all__ = [
    "MGDStars",
    "StarPlotter",
    "AnimationMaker",
]

# Package-level configuration
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())