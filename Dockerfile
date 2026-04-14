FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace Spaces runs containers as a non-root "user" with UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

COPY --chown=user:user requirements.txt ./
RUN pip install --user --upgrade pip && \
    pip install --user -r requirements.txt

COPY --chown=user:user . $HOME/app

# NHANES XPT files are not committed (HF blocks binaries in git). Fetch fresh at build time.
RUN mkdir -p data/nhanes && cd data/nhanes && \
    for f in BPXO_L.xpt DEMO_L.xpt BMX_L.xpt RXQ_RX_L.xpt; do \
        curl -sSfL --retry 3 -o "$f" "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/$f"; \
    done

# Train models at image build time so the container starts fast and is deterministic.
RUN python main.py || echo "Training failed at build time; will retry on startup via /api/train"

EXPOSE 7860

CMD ["sh", "-c", "test -f final_model/model.pkl || python main.py; uvicorn app:app --host 0.0.0.0 --port 7860"]
