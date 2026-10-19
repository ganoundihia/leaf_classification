"""Streamlit web app: upload a leaf photo and classify its margin."""

import hashlib
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
PROJECT_MODULE_DIRS = (APP_DIR, APP_DIR.parent / "src")


def forget_project_modules() -> None:
    """Drop the project's own modules from the import cache.

    Streamlit reruns this script after a code change but keeps imported modules
    in memory, so an updated specimen.py or leaf_classifier would be ignored
    until a full restart (on Streamlit Cloud, a push then raised ImportError).
    Only modules under app/ and src/ are dropped; libraries stay loaded.
    """
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if name == "__main__" or not file:
            continue
        path = Path(file).resolve()
        if any(path.is_relative_to(folder) for folder in PROJECT_MODULE_DIRS):
            del sys.modules[name]


forget_project_modules()

try:
    import leaf_classifier  # noqa: F401
except ModuleNotFoundError:
    # Package not pip-installed (e.g. on Streamlit Community Cloud): use the source tree
    sys.path.insert(0, str(APP_DIR.parent / "src"))

import leaf_classifier
from leaf_classifier import CLASS_NAMES, load_model
from leaf_classifier.leaf_gate import ClipLeafGate
from leaf_classifier.model import to_rgb
from leaf_classifier.ood import LeafDetector, analyze
from specimen import (
    MARGINS,
    accession_number,
    determination_label,
    margin_drawing,
    rejection_label,
)

HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.7
GITHUB_URL = "https://github.com/dihiaganoun/Leaf-Classification"
MODEL_URL = "https://huggingface.co/dganoun/leaf-classifier-resnext101"

st.set_page_config(
    page_title="Dentée ou lisse ? · Leaf Classifier",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def render_html(html: str) -> None:
    """Render HTML on one line: Markdown would turn indented lines into code blocks."""
    st.markdown(" ".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def package_version() -> str:
    """Hash of the leaf_classifier source files, so a deploy that changes them reloads the model."""
    digest = hashlib.sha1()
    for file in sorted(Path(leaf_classifier.__file__).parent.glob("*.py")):
        digest.update(file.read_bytes())
    return digest.hexdigest()


@st.cache_resource(show_spinner="Chargement des modèles…")
def get_models(version: str):  # noqa: ARG001 - only part of the cache key
    return load_model(), LeafDetector.load(), ClipLeafGate.load()


render_html(f"<style>{(APP_DIR / 'style.css').read_text(encoding='utf-8')}</style>")

try:
    model, detector, gate = get_models(package_version())
except (FileNotFoundError, RuntimeError, OSError) as err:
    st.error(f"Le modèle n'a pas pu être chargé : {err}")
    st.stop()

render_html(
    """
    <div class="eyebrow">Herbier numérique · Détermination du bord</div>
    <h1 class="headline">Dentée <em>ou</em> lisse&nbsp;?</h1>
    <p class="lede">Déposez la photo d'une feuille. Le modèle examine son bord
    et rend sa détermination sur une étiquette d'herbier.</p>
    """
)

uploaded_file = st.file_uploader(
    "Photo de la feuille",
    type=["jpg", "jpeg", "png"],
    help="JPG ou PNG. Une seule feuille, bien centrée, sur fond uni.",
)

if uploaded_file is None:
    render_html(
        '<div class="key">'
        + "".join(
            f"""
            <div class="key-item">
                {margin_drawing(name)}
                <div>
                    <div class="key-latin">{MARGINS[name]['latin']}</div>
                    <div class="key-common">Feuille {MARGINS[name]['label'].lower()}.<br>{MARGINS[name]['desc']}</div>
                </div>
            </div>
            """
            for name in CLASS_NAMES
        )
        + "</div>"
    )
else:
    data = uploaded_file.getvalue()
    image = to_rgb(Image.open(uploaded_file))

    with st.spinner("Examen du bord de la feuille…"):
        start = time.perf_counter()
        result = analyze(model, detector, image, gate)
        elapsed = time.perf_counter() - start

    col_specimen, col_label = st.columns([1, 1.2], gap="large")
    with col_specimen:
        st.image(image, width="stretch")
        render_html(
            f'<div class="specimen-caption">Spécimen · {uploaded_file.name} · '
            f"{image.width}×{image.height} px</div>"
        )
    with col_label:
        if result.is_leaf:
            label = determination_label(
                result.probs,
                accession_number(data),
                elapsed,
                HIGH_CONFIDENCE,
                LOW_CONFIDENCE,
                in_domain=result.in_domain,
            )
        else:
            label = rejection_label(result.rejection, accession_number(data), result.leaf_score)
        render_html(label)

render_html(
    f"""
    <div class="colophon">
        <span>ResNeXt-101 affiné · 98,68 % de précision sur 984 feuilles de test</span>
        <span><a href="{GITHUB_URL}" target="_blank">Code</a> ·
        <a href="{MODEL_URL}" target="_blank">Poids du modèle</a></span>
    </div>
    """
)
