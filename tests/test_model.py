"""Fast tests that do not need the real (LFS) weights."""

import pytest
import torch
import torch.nn as nn
from PIL import Image

from leaf_classifier import CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, load_model, predict, preprocess
from leaf_classifier import model as model_module
from leaf_classifier.model import to_rgb
from leaf_classifier.config import HF_REPO_ID, WEIGHTS_FILENAME
from leaf_classifier.data import stratified_split


class StubModel(nn.Module):
    """Returns fixed logits so predictions are deterministic."""

    def __init__(self, logits):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


@pytest.mark.parametrize("size", [(224, 224), (640, 480), (50, 300)])
def test_preprocess_shape(size):
    tensor = preprocess(Image.new("RGB", size, color=(30, 120, 40)))
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_converts_grayscale_and_rgba():
    for mode in ("L", "RGBA"):
        assert preprocess(Image.new(mode, (100, 100))).shape[1] == 3


def test_transparent_areas_become_white():
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    image.putpixel((5, 5), (20, 120, 40, 255))
    rgb = to_rgb(image)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (255, 255, 255)
    assert rgb.getpixel((5, 5)) == (20, 120, 40)


def test_predict_returns_probabilities_for_each_class():
    probs = predict(StubModel([2.0, -1.0]), Image.new("RGB", (300, 300)))
    assert list(probs) == CLASS_NAMES
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=probs.get) == "dente"


def test_class_config_is_consistent():
    assert len(CLASS_NAMES) == NUM_CLASSES == 2
    assert CLASS_NAMES == sorted(CLASS_NAMES), "must match ImageFolder alphabetical order"


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_model(tmp_path / "missing.pth")


def test_resolve_weights_prefers_local_copy(tmp_path, monkeypatch):
    local = tmp_path / "weights.pth"
    local.write_bytes(b"x")
    monkeypatch.setattr(model_module, "DEFAULT_MODEL_PATH", local)
    monkeypatch.setattr(model_module, "hf_hub_download", lambda **_: pytest.fail("should not download"))
    assert model_module.resolve_weights() == local


def test_resolve_weights_downloads_when_no_local_copy(tmp_path, monkeypatch):
    calls = {}

    def fake_download(repo_id, filename):
        calls.update(repo_id=repo_id, filename=filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(model_module, "DEFAULT_MODEL_PATH", tmp_path / "absent.pth")
    monkeypatch.setattr(model_module, "hf_hub_download", fake_download)
    assert model_module.resolve_weights() == tmp_path / WEIGHTS_FILENAME
    assert calls == {"repo_id": HF_REPO_ID, "filename": WEIGHTS_FILENAME}


def test_load_model_detects_lfs_pointer(tmp_path):
    pointer = tmp_path / "weights.pth"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(RuntimeError, match="LFS pointer"):
        load_model(pointer)


def test_stratified_split_is_disjoint_complete_and_reproducible():
    targets = [0] * 100 + [1] * 120
    split = stratified_split(targets)
    all_idx = split.train + split.val + split.test
    assert sorted(all_idx) == list(range(len(targets)))
    assert len(split.train) == 70 + 84
    assert len(split.val) == 15 + 18
    assert stratified_split(targets) == split
