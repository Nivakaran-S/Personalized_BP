"""
FastAPI backend for the BP personalized-alert system.

- Rule-based alert tier (CRISIS → NORMAL → CRITICAL_LOW) always evaluates first.
- ML prediction (supervised or per-patient IsoForest) adds personalized insight on top.
- Three-stakeholder messages (patient / caregiver / physician) per alert.
- `/api/predict` returns a structured alert object, not a single label.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from uvicorn import run as app_run

from src.constants import training_pipeline as C
from src.exception.exception import BPException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.main_utils.utils import load_object
from src.utils.ml_utils.alert_engine import (
    build_full_response,
    evaluate_alert_tier,
)
from src.utils.ml_utils.feature_engineering import predict_patient_alert

MODEL_STATE: Dict[str, Any] = {
    "supervised_bundle": None,
    "unsupervised_bundle": None,
}

# In-memory physician-target store (MVP — replaced by a real DB later).
PHYSICIAN_TARGETS: Dict[str, Dict[str, Any]] = {}


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


app = FastAPI(title="Healplace Cardio — BP Personalized Alert API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")


# ── Pydantic models ──────────────────────────────────────────────────────────

class Reading(BaseModel):
    systolic: float
    diastolic: float
    pulse: Optional[float] = None
    taken_at: Optional[str] = Field(default=None, description="ISO 8601 datetime or null")
    time_of_day: Optional[str] = Field(default=None, description="morning | afternoon | evening | night")
    position: Optional[str] = Field(default=None, description="sitting | standing | lying")


class PatientPayload(BaseModel):
    age: float = Field(..., description="Age in years")
    gender: int = Field(..., description="1=Male, 2=Female")
    bmi: Optional[float] = Field(default=None, description="Auto-computed from weight + height if omitted")
    weight: Optional[float] = None
    height: Optional[float] = None
    race_ethnicity: Optional[int] = Field(default=None, description="RIDRETH3 code")
    education: Optional[int] = Field(default=None, description="DMDEDUC2 code")
    income_ratio: Optional[float] = Field(default=None, description="INDFMPIR")
    antihypertensive_flag: int = Field(default=0, description="Legacy alias for on_antihypertensive")
    on_antihypertensive: Optional[int] = Field(default=None, description="Currently on BP medication")
    rx_count: float = 0

    # Clinical history
    has_diagnosed_htn: int = 0
    has_high_cholesterol: int = 0
    has_mi: int = 0
    has_stroke: int = 0
    has_heart_failure: int = 0
    has_chd: int = 0
    has_angina: int = 0
    has_diabetes: int = 0

    # Acute symptoms
    chest_pain_flag: int = 0
    severe_chest_pain_flag: int = 0
    sob_on_exertion_flag: int = 0
    dizziness_flag: int = Field(default=0, description="Feeling dizzy or lightheaded")

    # Optional patient ID for physician-target lookup
    patient_id: Optional[str] = None

    readings: List[Reading] = Field(..., min_length=1)


class PhysicianTarget(BaseModel):
    patient_id: str
    target_sys: float = 130
    target_dia: float = 80
    acceptable_range_sys: List[float] = Field(default=[120, 140])
    acceptable_range_dia: List[float] = Field(default=[70, 90])
    out_of_range_threshold: int = 3
    set_by: Optional[str] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "threshold": C.UNSUPERVISED_READINGS_THRESHOLD,
        "min_personalization": C.MIN_READINGS_FOR_PERSONALIZATION,
        "class_order": C.CLASS_ORDER,
    })


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "supervised_loaded": MODEL_STATE["supervised_bundle"] is not None,
        "unsupervised_loaded": MODEL_STATE["unsupervised_bundle"] is not None,
        "min_readings_for_personalization": C.MIN_READINGS_FOR_PERSONALIZATION,
        "unsupervised_threshold": C.UNSUPERVISED_READINGS_THRESHOLD,
        "class_order": C.CLASS_ORDER,
    }


@app.post("/api/predict")
async def api_predict(payload: PatientPayload) -> Dict[str, Any]:
    try:
        data = payload.model_dump()
        data["readings"] = [r.model_dump() for r in payload.readings]

        # Auto-compute BMI.
        if (data.get("bmi") is None) and data.get("weight") and data.get("height"):
            try:
                data["bmi"] = float(data["weight"]) / ((float(data["height"]) / 100.0) ** 2)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        # Unify antihypertensive fields.
        if data.get("on_antihypertensive") is None:
            data["on_antihypertensive"] = int(data.get("antihypertensive_flag", 0))
        data["antihypertensive_flag"] = int(data["on_antihypertensive"])

        # Determine the latest reading for the rule-based tier.
        latest = data["readings"][-1]
        sys_val = float(latest["systolic"])
        dia_val = float(latest["diastolic"])
        pulse_val = float(latest["pulse"]) if latest.get("pulse") else None

        symptoms = {
            "chest_pain_flag": data.get("chest_pain_flag", 0),
            "severe_chest_pain_flag": data.get("severe_chest_pain_flag", 0),
            "sob_on_exertion_flag": data.get("sob_on_exertion_flag", 0),
            "dizziness_flag": data.get("dizziness_flag", 0),
        }
        patient_context = {
            "on_antihypertensive": data.get("on_antihypertensive", 0),
            "has_diagnosed_htn": data.get("has_diagnosed_htn", 0),
            "has_mi": data.get("has_mi", 0),
            "has_stroke": data.get("has_stroke", 0),
            "has_heart_failure": data.get("has_heart_failure", 0),
            "has_chd": data.get("has_chd", 0),
            "has_angina": data.get("has_angina", 0),
            "has_diabetes": data.get("has_diabetes", 0),
            "high_risk_profile": int(
                bool(data.get("has_diagnosed_htn")) or bool(data.get("on_antihypertensive"))
                or bool(data.get("has_mi")) or bool(data.get("has_stroke"))
                or bool(data.get("has_heart_failure")) or bool(data.get("has_diabetes"))
            ),
        }

        # Physician target override (if set for this patient).
        physician_target = PHYSICIAN_TARGETS.get(data.get("patient_id")) if data.get("patient_id") else None

        # 1. Rule-based alert tier (always runs).
        alert = evaluate_alert_tier(sys_val, dia_val, pulse_val, symptoms, patient_context, physician_target)

        # 2. ML prediction (skipped if < MIN_READINGS_FOR_PERSONALIZATION).
        n = len(data["readings"])
        ml_prediction = None
        if n >= C.MIN_READINGS_FOR_PERSONALIZATION:
            ml_prediction = predict_patient_alert(
                data,
                MODEL_STATE["supervised_bundle"],
                MODEL_STATE["unsupervised_bundle"],
            )
            personalization_status = "active"
        else:
            personalization_status = "collecting_baseline"

        # 3. Assemble structured response.
        return build_full_response(
            alert=alert,
            ml_prediction=ml_prediction,
            personalization_status=personalization_status,
            n_readings=n,
            systolic=sys_val,
            diastolic=dia_val,
            pulse=pulse_val,
            symptoms=symptoms,
            patient_context=patient_context,
        )
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


# ── Physician target override (MVP in-memory store) ─────────────────────────

@app.post("/api/physician/target")
async def set_physician_target(target: PhysicianTarget) -> Dict[str, str]:
    PHYSICIAN_TARGETS[target.patient_id] = target.model_dump()
    return {"status": "ok", "patient_id": target.patient_id}


@app.get("/api/physician/target/{patient_id}")
async def get_physician_target(patient_id: str) -> Dict[str, Any]:
    t = PHYSICIAN_TARGETS.get(patient_id)
    if t is None:
        raise HTTPException(status_code=404, detail="No target set for this patient.")
    return t


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
