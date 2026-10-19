"""Herbarium-style visuals: leaf margin drawings and the determination label."""

import datetime as dt
import hashlib
import math

# Botanical vocabulary for each class (same order as CLASS_NAMES)
MARGINS = {
    "dente": {
        "latin": "Margo dentata",
        "label": "Dentée",
        "desc": "Bord découpé en petites dents.",
        "color": "#8A5A24",
    },
    "lisse": {
        "latin": "Margo integra",
        "label": "Lisse",
        "desc": "Bord continu, sans dents ni découpes.",
        "color": "#3D6B3A",
    },
}

ROMAN_MONTHS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def _leaf_path(toothed: bool, teeth: int = 13) -> str:
    """SVG path of a leaf blade (base at bottom, tip at top), smooth or serrated."""
    width, height, top, cx = 46.0, 170.0, 14.0, 60.0
    steps = 160
    right = []
    for i in range(steps + 1):
        s = i / steps  # 0 = base, 1 = tip
        half = width * math.sin(math.pi * s) ** 0.85 * (1 - 0.18 * s)
        if toothed and 0.04 < s < 0.96:
            phase = (s * teeth) % 1.0
            # Sawtooth: teeth lean toward the tip like a serrate margin
            half += 4.2 * (phase if phase < 0.8 else (1 - phase) * 4) * math.sin(math.pi * s) ** 0.5
        y = top + height * (1 - s)
        right.append((cx + half, y))
    left = [(2 * cx - x, y) for x, y in reversed(right)]
    points = right + left
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points) + " Z"
    return d


def margin_drawing(kind: str, state: str = "neutral") -> str:
    """Inline SVG of a leaf with the given margin. state: neutral | active | faded."""
    info = MARGINS[kind]
    stroke = info["color"] if state != "faded" else "#A9B2A6"
    fill = {"active": info["color"] + "1F", "neutral": "#FFFFFF", "faded": "#F4F6F0"}[state]
    base_y = 14 + 170
    return (
        f'<svg class="margin-svg {state}" viewBox="0 0 120 214" role="img" '
        f'aria-label="Feuille à bord {info["label"].lower()}">'
        f'<path d="{_leaf_path(kind == "dente")}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="1.6" stroke-linejoin="round"/>'
        f'<path d="M60 {base_y + 24} L60 22" stroke="{stroke}" stroke-width="1.4" fill="none"/>'
        + "".join(
            f'<path d="M60 {y} Q{60 + side * 18} {y - 10} {60 + side * 34} {y - 26}" '
            f'stroke="{stroke}" stroke-width="0.8" fill="none" opacity="0.7"/>'
            for y in (150, 122, 94, 66)
            for side in (-1, 1)
        )
        + "</svg>"
    )


def accession_number(data: bytes) -> str:
    return "LC-" + hashlib.sha1(data).hexdigest()[:6].upper()


def herbarium_date(day: dt.date | None = None) -> str:
    day = day or dt.date.today()
    return f"{day.day}.{ROMAN_MONTHS[day.month - 1]}.{day.year}"


def fr_percent(value: float) -> str:
    """French-formatted percentage that never rounds to a misleading 100 % or 0 %."""
    if value > 0.9999:
        return "> 99,99 %"
    if value < 0.0001:
        return "< 0,01 %"
    return f"{value * 100:.2f}".replace(".", ",") + " %"


def stamp_text(confidence: float, high: float, low: float) -> str:
    if confidence >= high:
        return "Déterminé"
    if confidence >= low:
        return "Probable"
    return "À vérifier"


REJECTIONS = {
    "not_leaf": (
        "Pas une feuille",
        "Aucune feuille n'est reconnue sur cette image. "
        "Envoyez la photo d'une seule feuille, à plat sur un fond uni.",
    ),
    "blank": (
        "Image vide",
        "L'image est presque uniforme : il n'y a rien à examiner. "
        "Envoyez la photo d'une feuille.",
    ),
}

OUT_OF_DOMAIN_NOTE = (
    "Cette photo ne ressemble pas aux planches d'herbier sur lesquelles le modèle "
    "a été entraîné : le résultat est indicatif."
)


def rejection_label(reason: str, accession: str, leaf_score: float | None = None) -> str:
    """HTML of the label shown when the image is not accepted as a leaf."""
    title, message = REJECTIONS[reason]
    score = (
        f"<dt>Feuille</dt><dd>{fr_percent(leaf_score)} selon CLIP</dd>"
        if reason == "not_leaf" and leaf_score is not None
        else ""
    )
    return f"""
    <div class="label-card label-rejected">
        <div class="label-head">
            <span>Herbier numérique</span>
            <span>N° {accession}</span>
        </div>
        <div class="label-text">
            <div class="label-latin">{title}</div>
            <p class="label-common">{message}</p>
            <dl class="label-fields">
                <dt>Det.</dt><dd>ResNeXt-101 affiné</dd>
                <dt>Date</dt><dd>{herbarium_date()}</dd>
                {score}
            </dl>
        </div>
        <div class="stamp stamp-doubt">
            <span class="stamp-word">Refusé</span>
        </div>
    </div>
    """


def determination_label(
    probs: dict[str, float],
    accession: str,
    elapsed_s: float,
    high: float,
    low: float,
    in_domain: bool = True,
) -> str:
    """HTML of the herbarium label holding the prediction.

    in_domain=False (the photo does not look like the training herbarium
    scans) stamps the result as indicative instead of determined.
    """
    pred = max(probs, key=probs.get)
    confidence = probs[pred]
    info = MARGINS[pred]
    stamp_word = stamp_text(confidence, high, low) if in_domain else "Indicatif"
    stamp_class = "stamp-doubt" if confidence < low else ("" if in_domain else "stamp-indicative")

    rows = "".join(
        f'<div class="prob-row">'
        f'<span class="prob-name">{MARGINS[name]["label"].lower()}</span>'
        f'<span class="prob-track"><span class="prob-fill" '
        f'style="width:{p * 100:.2f}%;background:{MARGINS[name]["color"]}"></span></span>'
        f'<span class="prob-value">{fr_percent(p)}</span>'
        f"</div>"
        for name, p in probs.items()
    )
    notes = []
    if not in_domain:
        notes.append(OUT_OF_DOMAIN_NOTE)
    if confidence < low:
        notes.append("Le modèle hésite. Reprenez une photo nette, une seule feuille sur fond uni.")
    doubt = "".join(f'<p class="label-doubt">{note}</p>' for note in notes)
    return f"""
    <div class="label-card">
        <div class="label-head">
            <span>Herbier numérique</span>
            <span>N° {accession}</span>
        </div>
        <div class="label-body">
            <div class="label-drawing">{margin_drawing(pred, "active")}</div>
            <div class="label-text">
                <div class="label-latin" style="color:{info['color']}">{info['latin']}</div>
                <div class="label-common">Feuille {info['label'].lower()} — {info['desc'].lower()}</div>
                <dl class="label-fields">
                    <dt>Det.</dt><dd>ResNeXt-101 affiné</dd>
                    <dt>Date</dt><dd>{herbarium_date()}</dd>
                    <dt>Durée</dt><dd>{f"{elapsed_s:.1f}".replace(".", ",")} s</dd>
                </dl>
            </div>
        </div>
        <div class="label-probs">{rows}</div>
        {doubt}
        <div class="stamp {stamp_class}">
            <span class="stamp-word">{stamp_word}</span>
            <span class="stamp-value">{fr_percent(confidence)}</span>
        </div>
    </div>
    """
