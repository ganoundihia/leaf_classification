"""Dataset splitting, identical to the split used in notebooks/training.ipynb."""

import math
import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def stratified_split(
    targets: list[int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> SplitIndices:
    """Per-class shuffled 70/15/15 split.

    Reproduces the training notebook exactly (same RNG calls in the same order),
    so it returns the same test set when the dataset folder has the same files.
    """
    idx_by_label = defaultdict(list)
    for idx, label in enumerate(targets):
        idx_by_label[label].append(idx)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for indices in idx_by_label.values():
        rng.shuffle(indices)
        n_train = int(math.floor(len(indices) * train_ratio))
        n_val = int(math.floor(len(indices) * val_ratio))
        train.extend(indices[:n_train])
        val.extend(indices[n_train:n_train + n_val])
        test.extend(indices[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitIndices(train=train, val=val, test=test)
