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

# NHANES XPT files are not committed (HF blocks binaries in git). Fetch fresh at build time
# and verify each download is a real XPT (not an HTML error page).
RUN set -eux; \
    mkdir -p data/nhanes; \
    cd data/nhanes; \
    for f in BPXO_L.xpt DEMO_L.xpt BMX_L.xpt RXQ_RX_L.xpt; do \
        echo "Downloading $f ..."; \
        curl -fL --retry 5 --retry-delay 3 --connect-timeout 30 --max-time 300 \
             -A "Mozilla/5.0 (HF-Space build)" \
             -o "$f" "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/$f"; \
        size=$(stat -c%s "$f"); \
        echo "  -> $f is $size bytes"; \
        if [ "$size" -lt 100000 ]; then \
            echo "ERROR: $f is suspiciously small — likely an error page, not an XPT file."; \
            head -c 200 "$f"; echo; \
            exit 1; \
        fi; \
        head -c 8 "$f" | od -c | head -1; \
    done; \
    ls -l

# Train models at image build time so the container starts fast and is deterministic.
RUN python main.py

EXPOSE 7860

CMD ["sh", "-c", "test -f final_model/model.pkl || python main.py; uvicorn app:app --host 0.0.0.0 --port 7860"]
