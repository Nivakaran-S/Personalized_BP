"""
FastAPI backend for the BP personalized-alert system.

- Single-page UI at `/` (templates/index.html) that calls JSON APIs via fetch.
- `/api/predict` routes to the supervised or unsupervised model based on reading count.
- `/api/train` kicks off the training pipeline.
- `/api/health` reports model-load status.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from uvicorn import run as app_run

from src.constants import training_pipeline as C
from src.exception.exception import BPException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object
from src.utils.ml_utils.feature_engineering import predict_patient_alert


MODEL_STATE: Dict[str, Any] = {
    "supervised_bundle": None,
    "unsupervised_bundle": None,
}


def _load_models() -> None:
    base = C.SAVED_MODEL_DIR
    model_path = os.path.join(base, C.MODEL_FILE_NAME)
    pre_path = os.path.join(base, C.PREPROCESSOR_FILE_NAME)
    unsup_path = os.path.join(base, C.UNSUPERVISED_FILE_NAME)
    meta_path = os.path.join(base, C.FEATURE_METADATA_FILE_NAME)

    if all(os.path.exists(p) for p in [model_path, pre_path, meta_path]):
        with open(meta_path) as f:
            meta = json.load(f)
        MODEL_STATE["supervised_bundle"] = {
            "model": load_object(model_path),
            "preprocessor": load_object(pre_path),
            "feature_metadata": meta,
        }
        logging.info("Loaded supervised bundle from %s", base)
    else:
        logging.warning("Supervised model files missing under %s — run /api/train first.", base)

    if os.path.exists(unsup_path):
        MODEL_STATE["unsupervised_bundle"] = load_object(unsup_path)
        logging.info("Loaded unsupervised bundle from %s", unsup_path)
    else:
        logging.warning("Unsupervised bundle missing at %s.", unsup_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_models()
    except Exception as exc:  # noqa: BLE001
        logging.warning("Model load failed at startup: %s", exc)
    yield


app = FastAPI(title="BP Personalized Alert API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")


class Reading(BaseModel):
    systolic: float
    diastolic: float
    pulse: Optional[float] = None


class PatientPayload(BaseModel):
    age: float = Field(..., description="RIDAGEYR — age in years")
    gender: int = Field(..., description="RIAGENDR (1=Male, 2=Female)")
    bmi: Optional[float] = Field(default=None, description="Auto-computed from weight + height if omitted")
    weight: Optional[float] = None
    height: Optional[float] = None
    race_ethnicity: Optional[int] = Field(default=None, description="RIDRETH3 code")
    education: Optional[int] = Field(default=None, description="DMDEDUC2 code")
    income_ratio: Optional[float] = Field(default=None, description="INDFMPIR")
    antihypertensive_flag: int = Field(default=0, description="Legacy alias for on_antihypertensive")
    on_antihypertensive: Optional[int] = Field(default=None, description="BPQ050A: currently on BP medication")
    rx_count: float = 0

    # Clinical history (BPQ, MCQ, DIQ)
    has_diagnosed_htn: int = Field(default=0, description="BPQ020: ever told had high BP")
    has_high_cholesterol: int = Field(default=0, description="BPQ080: ever told had high cholesterol")
    has_mi: int = Field(default=0, description="MCQ160E: ever had heart attack")
    has_stroke: int = Field(default=0, description="MCQ160F: ever had stroke")
    has_heart_failure: int = Field(default=0, description="MCQ160B: ever had heart failure")
    has_chd: int = Field(default=0, description="MCQ160C: coronary heart disease")
    has_angina: int = Field(default=0, description="MCQ160D: angina/angina pectoris")
    has_diabetes: int = Field(default=0, description="DIQ010: ever told had diabetes")

    # Acute symptoms (CDQ)
    chest_pain_flag: int = Field(default=0, description="CDQ001: chest pain or discomfort")
    severe_chest_pain_flag: int = Field(default=0, description="CDQ008: severe or half-hour chest pain")
    sob_on_exertion_flag: int = Field(default=0, description="CDQ010: shortness of breath on stairs")

    readings: List[Reading] = Field(..., min_length=1)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "threshold": C.UNSUPERVISED_READINGS_THRESHOLD,
            "class_order": C.CLASS_ORDER,
        },
    )


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "supervised_loaded": MODEL_STATE["supervised_bundle"] is not None,
        "unsupervised_loaded": MODEL_STATE["unsupervised_bundle"] is not None,
        "unsupervised_threshold": C.UNSUPERVISED_READINGS_THRESHOLD,
        "class_order": C.CLASS_ORDER,
    }


@app.post("/api/predict")
async def api_predict(payload: PatientPayload) -> Dict[str, Any]:
    if MODEL_STATE["supervised_bundle"] is None and MODEL_STATE["unsupervised_bundle"] is None:
        raise HTTPException(status_code=503, detail="No models loaded. Call /api/train first.")
    try:
        data = payload.model_dump()
        data["readings"] = [r.model_dump() for r in payload.readings]
        # Auto-compute BMI from weight + height if the client didn't send it.
        if (data.get("bmi") is None) and data.get("weight") and data.get("height"):
            try:
                data["bmi"] = float(data["weight"]) / ((float(data["height"]) / 100.0) ** 2)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        # Unify the two antihypertensive fields so downstream code sees one source of truth.
        if data.get("on_antihypertensive") is None:
            data["on_antihypertensive"] = int(data.get("antihypertensive_flag", 0))
        data["antihypertensive_flag"] = int(data["on_antihypertensive"])
        result = predict_patient_alert(
            data,
            MODEL_STATE["supervised_bundle"],
            MODEL_STATE["unsupervised_bundle"],
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BPException as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/train")
async def api_train() -> Dict[str, Any]:
    try:
        artifact = TrainingPipeline().run_pipeline()
        _load_models()
        return {
            "status": "ok",
            "best_model_name": artifact.best_model_name,
            "test_metrics": artifact.test_metrics.__dict__,
            "train_metrics": artifact.train_metrics.__dict__,
            "model_bundle_file_path": artifact.model_bundle_file_path,
        }
    except BPException as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
