"""Central configuration shared by the app, the evaluation script and the tests."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Class order must match the training ImageFolder order (alphabetical folder names).
CLASS_NAMES = ["dente", "lisse"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WEIGHTS_FILENAME = "resnext101_leaves.pth"

# Local copy, used first if present. Override with LEAF_MODEL_PATH.
DEFAULT_MODEL_PATH = Path(
    os.environ.get("LEAF_MODEL_PATH", PROJECT_ROOT / "models" / WEIGHTS_FILENAME)
)

# Hugging Face Hub repository the weights are downloaded from when no local copy exists.
HF_REPO_ID = os.environ.get("LEAF_HF_REPO", "dganoun/leaf-classifier-resnext101")
