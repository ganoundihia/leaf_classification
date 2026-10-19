FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# CPU-only PyTorch keeps the image much smaller than the default CUDA build
RUN pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu

# Run as a non-root user (also required by Hugging Face Spaces)
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user pyproject.toml README.md LICENSE requirements.txt ./
COPY --chown=user src ./src
RUN pip install --user -r requirements.txt

# Download the weights, the leaf detector and CLIP at build time so the app starts without waiting
RUN python -c "from leaf_classifier.model import resolve_weights; from leaf_classifier.ood import LeafDetector; from leaf_classifier.leaf_gate import ClipLeafGate; print(resolve_weights()); LeafDetector.load(); ClipLeafGate.load()"

COPY --chown=user .streamlit ./.streamlit
COPY --chown=user app ./app

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
