"""Measure the CLIP "is this a leaf?" check on leaves and on other images.

The threshold (leaf_gate.LEAF_THRESHOLD) is fixed in advance, not tuned on
these sets, so the reported rates are measurements.

Example:
    python scripts/calibrate_leaf_gate.py --data-dir path/to/feuilles_plantes \\
        --leaf-dir path/to/fresh_leaves --ood-dir path/to/imagenette
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder

from leaf_classifier.config import PROJECT_ROOT
from leaf_classifier.data import stratified_split
from leaf_classifier.leaf_gate import ClipLeafGate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="Herbarium dataset.")
    parser.add_argument("--herbarium-per-class", type=int, default=150)
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Herbarium split to measure on (use val while developing).")
    parser.add_argument("--leaf-dir", type=Path, action="append", default=[],
                        help="Folder of other leaf photos (searched recursively).")
    parser.add_argument("--ood-dir", type=Path, action="append", default=[],
                        help="Folder of non-leaf images (searched recursively).")
    parser.add_argument("--max-per-dir", type=int, default=500)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "docs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def list_images(folder: Path, limit: int, rng: random.Random) -> list[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(rng.sample(files, min(limit, len(files))))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    gate = ClipLeafGate.load()

    dataset = ImageFolder(args.data_dir)
    split_idx = getattr(stratified_split(dataset.targets), args.split)
    herbarium = []
    for label in range(len(dataset.classes)):
        idx = [i for i in split_idx if dataset.targets[i] == label]
        herbarium += [Path(dataset.samples[i][0]) for i in rng.sample(idx, args.herbarium_per_class)]

    groups = {f"herbarium {args.split} leaves": ("leaf", herbarium)}
    for folder in args.leaf_dir:
        groups[f"leaves: {folder.name}"] = ("leaf", list_images(folder, args.max_per_dir, rng))
    for folder in args.ood_dir:
        groups[f"not leaves: {folder.name}"] = ("other", list_images(folder, args.max_per_dir, rng))

    report = {"threshold": gate.threshold, "groups": {}}
    scores = {}
    for name, (kind, paths) in groups.items():
        s = np.array([gate.leaf_score(Image.open(p)) for p in paths])
        scores[name] = s
        accepted = float(np.mean(s >= gate.threshold))
        rate = {"accepted": round(accepted, 4)} if kind == "leaf" else {"rejected": round(1 - accepted, 4)}
        wrong = [p.name for p, v in zip(paths, s) if (v >= gate.threshold) != (kind == "leaf")]
        report["groups"][name] = {"n": len(s), **rate, "errors": wrong[:20]}
        print(f"{name:32s} n={len(s):4d} {rate}", flush=True)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "val":
        return  # development run: keep the published report untouched
    (args.report_dir / "leaf_gate.json").write_text(json.dumps(report, indent=2))

    plt.figure(figsize=(7, 3.6))
    bins = np.linspace(0, 1, 41)
    for name, s in scores.items():
        plt.hist(s, bins=bins, alpha=0.55, label=f"{name} (n={len(s)})")
    plt.axvline(gate.threshold, color="#5A3E8E", linestyle="--", label="threshold")
    plt.yscale("log")
    plt.xlabel("CLIP probability that the image is a leaf")
    plt.ylabel("images (log scale)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.report_dir / "leaf_gate.png", dpi=120)


if __name__ == "__main__":
    main()
