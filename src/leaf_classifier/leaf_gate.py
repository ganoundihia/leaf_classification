"""Zero-shot "is this a leaf?" check with CLIP.

The Mahalanobis detector (ood.py) only knows the look of the training
herbarium scans: it rejects real photos of fresh leaves and can accept
objects that share that look. CLIP compares the image with text
descriptions instead, so it judges what the image shows, not its style.
"""

from dataclasses import dataclass

import torch
from PIL import Image

from leaf_classifier.model import to_rgb

CLIP_MODEL = "ViT-B-32"
CLIP_WEIGHTS = "laion2b_s34b_b79k"

LEAF_PROMPTS = [
    "a photo of a leaf",
    "a photo of a single green leaf",
    "a close-up photo of a plant leaf",
    "a herbarium specimen of a pressed leaf",
    "a scan of a dried leaf on paper",
]
OTHER_PROMPTS = [
    "a photo of food",
    "a photo of fruit",
    "a photo of berries",
    "a photo of a dessert",
    "a photo of a cake",
    "a photo of an animal",
    "a photo of a person",
    "a photo of a vehicle",
    "a photo of a building",
    "a screenshot",
    "a diagram",
    "a photo of an everyday object",
    "a photo of a flower",
    "a photo of a whole tree",
    "a photo of a landscape",
    "a texture",
    "a plain colored background",
]

# Minimum share of the probability on the leaf descriptions. Chosen on
# development data (herbarium validation split + 500 Imagenette photos not used
# in the report): 99.0% of leaves accepted, 99.8% of photos rejected.
# Measured on held-out data by scripts/calibrate_leaf_gate.py (docs/leaf_gate.json).
LEAF_THRESHOLD = 0.2


@dataclass
class ClipLeafGate:
    visual: torch.nn.Module
    preprocess: object
    text_features: torch.Tensor  # (n_prompts, d), L2-normalised
    logit_scale: float
    threshold: float = LEAF_THRESHOLD

    @classmethod
    def load(cls, threshold: float = LEAF_THRESHOLD) -> "ClipLeafGate":
        import open_clip  # imported here: heavy, and only needed when the gate is used

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_WEIGHTS
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        with torch.no_grad():
            text = model.encode_text(tokenizer(LEAF_PROMPTS + OTHER_PROMPTS))
            text = text / text.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp().item()
        # Keep only the image tower: the text features are computed once
        return cls(model.visual, preprocess, text, logit_scale, threshold)

    @torch.no_grad()
    def leaf_score(self, image: Image.Image) -> float:
        """Probability mass CLIP puts on the leaf descriptions (0 to 1)."""
        features = self.visual(self.preprocess(to_rgb(image)).unsqueeze(0))
        features = features / features.norm(dim=-1, keepdim=True)
        probs = (self.logit_scale * features @ self.text_features.T).softmax(dim=-1)[0]
        return float(probs[: len(LEAF_PROMPTS)].sum())

    def is_leaf(self, image: Image.Image) -> bool:
        return self.leaf_score(image) >= self.threshold
