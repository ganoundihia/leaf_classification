"""Reject images that are not leaves (out-of-distribution detection).

The classifier always answers "dente" or "lisse", even for a screenshot. Before
trusting that answer, we check that the image's 2048-d feature vector (the
layer just before the classifier) is close to the feature vectors of the
training leaves, using the Mahalanobis distance to the nearest class mean with
a shared covariance (Lee et al., 2018). No extra model is needed: the features
come from the same forward pass as the prediction.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image, ImageStat

from leaf_classifier.config import CLASS_NAMES, HF_REPO_ID, PROJECT_ROOT
from leaf_classifier.model import preprocess, to_rgb

OOD_FILENAME = "ood_stats.npz"
DEFAULT_OOD_PATH = PROJECT_ROOT / "models" / OOD_FILENAME


@torch.no_grad()
def forward_features(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (penultimate features, logits) for a torchvision ResNet/ResNeXt."""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    features = torch.flatten(model.avgpool(x), 1)
    return features, model.fc(features)


@dataclass
class LeafDetector:
    class_means: np.ndarray  # (n_classes, d)
    whitening: np.ndarray  # (d, d), W with W @ W.T = inverse of the shared covariance
    threshold: float  # distances above this are rejected

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, covariance: np.ndarray) -> "LeafDetector":
        """Build from training features and their shared (class-centred) covariance.

        Storing the whitening matrix instead of the precision matrix keeps the
        distances accurate in float32: its condition number is the square root
        of the precision's.
        """
        features = features.astype(np.float64)
        means = np.stack([features[labels == c].mean(0) for c in np.unique(labels)])
        eigvals, eigvecs = np.linalg.eigh(covariance.astype(np.float64))
        whitening = eigvecs / np.sqrt(np.maximum(eigvals, 1e-12))
        return cls(class_means=means, whitening=whitening, threshold=np.inf)

    def distance(self, features: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each feature vector to the nearest class mean."""
        features = np.atleast_2d(features).astype(np.float64)
        dists = [np.linalg.norm((features - mean) @ self.whitening, axis=1) for mean in self.class_means]
        return np.min(dists, axis=0)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            class_means=self.class_means.astype(np.float32),
            whitening=self.whitening.astype(np.float32),
            threshold=np.float64(self.threshold),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LeafDetector":
        if path is None:
            path = DEFAULT_OOD_PATH if DEFAULT_OOD_PATH.is_file() else hf_hub_download(
                repo_id=HF_REPO_ID, filename=OOD_FILENAME
            )
        data = np.load(path)
        return cls(
            class_means=data["class_means"].astype(np.float64),
            whitening=data["whitening"].astype(np.float64),
            threshold=float(data["threshold"]),
        )


# Pixel standard deviation (0-255 grey levels) below which an image counts as blank.
# The least contrasted leaf in the dataset has 19.2.
MIN_PIXEL_STD = 5.0


@dataclass
class Analysis:
    probs: dict[str, float]
    distance: float  # Mahalanobis distance to the training leaves
    in_domain: bool  # looks like the training herbarium scans
    leaf_score: float | None  # CLIP probability that the image is a leaf (None without gate)
    rejection: str | None  # None, "blank" or "not_leaf"

    @property
    def is_leaf(self) -> bool:
        return self.rejection is None


def analyze(model: nn.Module, detector: LeafDetector, image: Image.Image, gate=None) -> Analysis:
    """Classify the image and decide whether the answer can be trusted.

    With a CLIP gate (leaf_gate.ClipLeafGate), CLIP decides whether the image
    is a leaf and the Mahalanobis detector only reports whether it looks like
    the training herbarium scans. Without one, the detector decides alone.
    """
    device = next(model.parameters()).device
    features, logits = forward_features(model, preprocess(image).to(device))
    probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    distance = float(detector.distance(features.cpu().numpy())[0])
    in_domain = distance <= detector.threshold
    leaf_score = None

    if ImageStat.Stat(to_rgb(image).convert("L")).stddev[0] < MIN_PIXEL_STD:
        rejection = "blank"
    elif gate is not None:
        leaf_score = gate.leaf_score(image)
        rejection = None if leaf_score >= gate.threshold else "not_leaf"
    else:
        rejection = None if in_domain else "not_leaf"
    return Analysis(
        probs=dict(zip(CLASS_NAMES, probs)),
        distance=distance,
        in_domain=in_domain,
        leaf_score=leaf_score,
        rejection=rejection,
    )
