"""Evaluate the model on a stratified sample of the held-out test split.

Example:
    python scripts/evaluate.py --data-dir path/to/feuilles_plantes --per-class 150
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torchvision.datasets import ImageFolder

from leaf_classifier import CLASS_NAMES, load_model, predict
from leaf_classifier.config import PROJECT_ROOT
from leaf_classifier.data import stratified_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Folder containing one sub-folder per class.")
    parser.add_argument("--per-class", type=int, default=150,
                        help="Number of test images sampled per class.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "docs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ImageFolder(args.data_dir)
    if dataset.classes != CLASS_NAMES:
        raise ValueError(f"Expected classes {CLASS_NAMES}, found {dataset.classes}")

    test_idx = stratified_split(dataset.targets).test
    rng = random.Random(args.seed)
    sample = []
    for label in range(len(CLASS_NAMES)):
        class_idx = [i for i in test_idx if dataset.targets[i] == label]
        sample += rng.sample(class_idx, min(args.per_class, len(class_idx)))

    model = load_model()
    y_true, y_pred = [], []
    start = time.perf_counter()
    for n, i in enumerate(sample, 1):
        path, label = dataset.samples[i]
        probs = predict(model, Image.open(path))
        y_true.append(label)
        y_pred.append(CLASS_NAMES.index(max(probs, key=probs.get)))
        if n % 25 == 0:
            print(f"{n}/{len(sample)} images", flush=True)
    elapsed = time.perf_counter() - start

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    metrics = {
        "n_images": len(sample),
        "per_class": args.per_class,
        "seed": args.seed,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "seconds_per_image": round(elapsed / len(sample), 3),
    }
    print(report)
    print(json.dumps(metrics, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics_sample.json").write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES
    ).plot(cmap="Greens", values_format="d", colorbar=False)
    plt.title(f"Confusion matrix (test sample, n={len(sample)})")
    plt.tight_layout()
    plt.savefig(args.out_dir / "confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()
