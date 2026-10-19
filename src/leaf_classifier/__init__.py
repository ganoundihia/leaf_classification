"""Leaf margin classification (toothed vs smooth) with a fine-tuned ResNeXt-101."""

from leaf_classifier.config import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES
from leaf_classifier.model import build_model, load_model, predict, preprocess

__all__ = [
    "CLASS_NAMES",
    "IMAGE_SIZE",
    "NUM_CLASSES",
    "build_model",
    "load_model",
    "predict",
    "preprocess",
]
__version__ = "1.0.0"
