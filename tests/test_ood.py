"""Tests for the "is this a leaf?" detector (no real weights needed)."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import models

from leaf_classifier.ood import LeafDetector, analyze, forward_features


def make_detector(dim=4, threshold=3.0):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, dim))
    labels = np.array([0] * 100 + [1] * 100)
    features[labels == 1] += 5.0
    centred = features - np.stack([features[labels == c].mean(0) for c in (0, 1)])[labels]
    detector = LeafDetector.fit(features, labels, np.cov(centred, rowvar=False))
    detector.threshold = threshold
    return detector, features, labels, centred


def test_distance_matches_explicit_mahalanobis():
    detector, features, labels, centred = make_detector()
    precision = np.linalg.inv(np.cov(centred, rowvar=False))
    x = np.array([[0.5, -1.0, 2.0, 0.0]])
    expected = min(
        float(np.sqrt(((x - m) @ precision @ (x - m).T).item())) for m in detector.class_means
    )
    assert detector.distance(x)[0] == pytest.approx(expected, rel=1e-6)


def test_far_points_are_farther_than_training_points():
    detector, features, *_ = make_detector()
    assert np.median(detector.distance(features)) < detector.distance(np.full((1, 4), 50.0))[0]


def test_save_and_load_round_trip(tmp_path):
    detector, features, *_ = make_detector()
    detector.save(tmp_path / "ood.npz")
    loaded = LeafDetector.load(tmp_path / "ood.npz")
    assert loaded.threshold == detector.threshold
    np.testing.assert_allclose(loaded.distance(features), detector.distance(features), rtol=1e-4)


@pytest.fixture(scope="module")
def tiny_resnet():
    torch.manual_seed(0)
    return models.resnet18(weights=None, num_classes=2).eval()


def test_forward_features_matches_model_output(tiny_resnet):
    x = torch.randn(2, 3, 224, 224)
    features, logits = forward_features(tiny_resnet, x)
    assert features.shape == (2, 512)
    torch.testing.assert_close(logits, tiny_resnet(x))


def leaf_like_image():
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (120, 120, 3), dtype=np.uint8))


def test_analyze_accepts_and_rejects_by_threshold(tiny_resnet):
    detector = LeafDetector(np.zeros((2, 512)), np.eye(512), threshold=np.inf)
    accepted = analyze(tiny_resnet, detector, leaf_like_image())
    assert accepted.is_leaf and accepted.rejection is None
    assert sum(accepted.probs.values()) == pytest.approx(1.0)

    detector.threshold = 0.0
    assert analyze(tiny_resnet, detector, leaf_like_image()).rejection == "not_leaf"


class FakeGate:
    def __init__(self, score, threshold=0.2):
        self.score, self.threshold = score, threshold

    def leaf_score(self, image):
        return self.score


def test_gate_decides_when_given(tiny_resnet):
    detector = LeafDetector(np.zeros((2, 512)), np.eye(512), threshold=np.inf)
    accepted = analyze(tiny_resnet, detector, leaf_like_image(), FakeGate(0.9))
    assert accepted.is_leaf and accepted.leaf_score == 0.9 and accepted.in_domain

    rejected = analyze(tiny_resnet, detector, leaf_like_image(), FakeGate(0.05))
    assert rejected.rejection == "not_leaf"


def test_out_of_domain_leaf_is_accepted_but_flagged(tiny_resnet):
    detector = LeafDetector(np.zeros((2, 512)), np.eye(512), threshold=0.0)
    result = analyze(tiny_resnet, detector, leaf_like_image(), FakeGate(0.9))
    assert result.is_leaf
    assert not result.in_domain


def test_clip_gate_score_from_prompt_similarity():
    from leaf_classifier.leaf_gate import LEAF_PROMPTS, OTHER_PROMPTS, ClipLeafGate

    n_prompts = len(LEAF_PROMPTS) + len(OTHER_PROMPTS)
    text = torch.eye(n_prompts)  # one orthogonal unit vector per prompt

    def gate_seeing(prompt_index):
        visual = lambda x: text[prompt_index].unsqueeze(0)  # noqa: E731
        return ClipLeafGate(visual, lambda image: torch.zeros(3, 2, 2), text, logit_scale=100.0)

    assert gate_seeing(0).leaf_score(leaf_like_image()) > 0.99  # matches a leaf prompt
    assert gate_seeing(len(LEAF_PROMPTS)).leaf_score(leaf_like_image()) < 0.01  # matches "food"
    assert not gate_seeing(len(LEAF_PROMPTS)).is_leaf(leaf_like_image())


def test_analyze_rejects_blank_image(tiny_resnet):
    detector = LeafDetector(np.zeros((2, 512)), np.eye(512), threshold=np.inf)
    result = analyze(tiny_resnet, detector, Image.new("RGB", (100, 100), (216, 162, 130)))
    assert result.rejection == "blank"
