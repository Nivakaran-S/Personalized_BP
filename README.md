---
title: BP Personalized Alert
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# BP Personalized Alert

Personalized blood-pressure classification (Hypotensive / Normal / Hypertensive) trained on NHANES 2021 data. Mirrors the ML design in [notebooks/experiment_08.ipynb](notebooks/experiment_08.ipynb).

- **Supervised path** (≤ 15 readings): RandomForest / ExtraTrees / Logistic Regression / XGBoost, selected by 5-fold CV macro-F1.
- **Unsupervised path** (> 15 readings): patient-baseline z-score rule backed by a population IsolationForest.

## Run locally

```
pip install -r requirements.txt
python main.py            # train
uvicorn app:app --reload  # serve UI at http://localhost:8000/
```

## Endpoints

- `GET /` — web UI
- `POST /api/predict` — JSON patient payload → class + probabilities
- `POST /api/train` — runs the full training pipeline
- `GET /api/health` — model-load status

## Deployment

Pushed to HuggingFace Spaces via [.github/workflows/deploy-huggingface.yml](.github/workflows/deploy-huggingface.yml). Training runs inside the Docker image build against the committed NHANES XPT files under [data/nhanes/](data/nhanes/).
