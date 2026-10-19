"""Fit the "is this a leaf?" detector, calibrate its threshold and evaluate it.

1. Extract features on a sample of the train / val / test splits (cached).
2. Fit class means and a shared covariance (Ledoit-Wolf shrinkage) on train.
3. Set the threshold so that --keep of the validation leaves are accepted.
4. Report the acceptance rate on test leaves and the rejection rate on
   folders of non-leaf images (--ood-dir, can be repeated).

Example:
    python scripts/fit_leaf_detector.py --data-dir path/to/feuilles_plantes \\
        --ood-dir path/to/imagenette/val --ood-dir path/to/screenshots
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf
from torchvision.datasets import ImageFolder

from leaf_classifier import CLASS_NAMES, load_model, preprocess
from leaf_classifier.config import PROJECT_ROOT
from leaf_classifier.data import stratified_split
from leaf_classifier.ood import LeafDetector, forward_features

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ood-dir", type=Path, action="append", default=[],
                        help="Folder of non-leaf images (searched recursively).")
    parser.add_argument("--ood-max", type=int, default=500, help="Max images per OOD folder.")
    parser.add_argument("--train-per-class", type=int, default=1500)
    parser.add_argument("--eval-per-class", type=int, default=300)
    parser.add_argument("--keep", type=float, default=0.99,
                        help="Share of validation leaves the threshold must accept.")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "features")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "ood_stats.npz")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "docs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def extract(model, paths: list[Path], cache: Path, batch_size: int = 16) -> np.ndarray:
    """Penultimate features for each image, cached on disk."""
    if cache.is_file():
        return np.load(cache)["features"]
    device = next(model.parameters()).device
    feats = []
    for start in range(0, len(paths), batch_size):
        batch = torch.cat([preprocess(Image.open(p)) for p in paths[start:start + batch_size]])
        feats.append(forward_features(model, batch.to(device))[0].cpu().numpy())
        print(f"  {cache.stem}: {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    feats = np.concatenate(feats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, features=feats)
    return feats


def sample_split(dataset: ImageFolder, indices: list[int], per_class: int, rng: random.Random):
    paths, labels = [], []
    for label in range(len(CLASS_NAMES)):
        class_idx = [i for i in indices if dataset.targets[i] == label]
        for i in sorted(rng.sample(class_idx, min(per_class, len(class_idx)))):
            paths.append(Path(dataset.samples[i][0]))
            labels.append(label)
    return paths, np.array(labels)


def list_images(folder: Path, limit: int, rng: random.Random) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(rng.sample(files, min(limit, len(files))))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    dataset = ImageFolder(args.data_dir)
    split = stratified_split(dataset.targets)
    model = load_model()

    train_paths, train_labels = sample_split(dataset, split.train, args.train_per_class, rng)
    val_paths, _ = sample_split(dataset, split.val, args.eval_per_class, rng)
    test_paths, _ = sample_split(dataset, split.test, args.eval_per_class, rng)

    print("Extracting features (cached after the first run)")
    train = extract(model, train_paths, args.cache_dir / f"train_{args.train_per_class}.npz")
    val = extract(model, val_paths, args.cache_dir / f"val_{args.eval_per_class}.npz")
    test = extract(model, test_paths, args.cache_dir / f"test_{args.eval_per_class}.npz")

    # Shared covariance of the class-centred training features, in float64:
    # float32 makes the inverse of this 2048x2048 matrix inaccurate.
    train = train.astype(np.float64)
    class_means = np.stack([train[train_labels == c].mean(0) for c in range(len(CLASS_NAMES))])
    covariance = LedoitWolf().fit(train - class_means[train_labels]).covariance_

    detector = LeafDetector.fit(train, train_labels, covariance)
    val_d = detector.distance(val)
    detector.threshold = float(np.quantile(val_d, args.keep))
    test_d = detector.distance(test)

    report = {
        "threshold": round(detector.threshold, 3),
        "keep_target_on_val": args.keep,
        "test_leaves": {"n": len(test_d), "accepted": round(float(np.mean(test_d <= detector.threshold)), 4)},
        "non_leaves": {},
    }
    ood_distances = {}
    for folder in args.ood_dir:
        paths = list_images(folder, args.ood_max, rng)
        feats = extract(model, paths, args.cache_dir / f"ood_{folder.name}_{len(paths)}.npz")
        d = detector.distance(feats)
        ood_distances[folder.name] = d
        report["non_leaves"][folder.name] = {
            "n": len(d), "rejected": round(float(np.mean(d > detector.threshold)), 4)
        }

    print(json.dumps(report, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    detector.save(args.out)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "leaf_detector.json").write_text(json.dumps(report, indent=2))

    plt.figure(figsize=(7, 3.6))
    upper = max(np.max(test_d), *[np.percentile(d, 95) for d in ood_distances.values()], 1)
    bins = np.linspace(0, upper, 60)
    plt.hist(np.clip(test_d, 0, upper), bins=bins, alpha=0.8, label="test leaves", color="#3D6B3A")
    for (name, d), color in zip(ood_distances.items(), ["#C98A3A", "#5B8DB8", "#9A9A9A", "#B5526B"]):
        plt.hist(np.clip(d, 0, upper), bins=bins, alpha=0.6, label=f"not leaves: {name}", color=color)
    plt.axvline(detector.threshold, color="#5A3E8E", linestyle="--", label="threshold")
    plt.title("Distance to the training leaves (values beyond the axis are clipped)", fontsize=10)
    plt.xlabel("Mahalanobis distance to the training leaves")
    plt.ylabel("images")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.report_dir / "leaf_detector.png", dpi=120)


if __name__ == "__main__":
    main()
