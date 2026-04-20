"""
FastAPI backend for HealPlace Cardio (rule-based).

No ML. All alert logic is rule-based, driven by Dr. Manisha Singal's signed-off
clinical specification. See `src/alert_engine.py` + `src/constants/alert_rules.py`.

Endpoints:
- GET  /                             → single-page UI
- GET  /api/health                   → liveness
- POST /api/predict                  → evaluate a PatientPayload, return alert + messages
- POST /api/clinical_context         → return the thresholds that WOULD apply (no reading needed)
- POST /api/physician/target         → upsert a per-patient threshold override (in-memory MVP)
- GET  /api/physician/target/{pid}   → fetch a patient's override
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from uvicorn import run as app_run

from src.alert_engine import evaluate, result_to_dict, select_thresholds
from src.constants import alert_rules as R
from src.exception.exception import BPException
from src.logging.logger import logging


# ---------------------------------------------------------------------------
# App + CORS + templates
# ---------------------------------------------------------------------------

app = FastAPI(title="HealPlace Cardio Alert API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# In-memory physician-target store (MVP — Cardioplace will persist this)
# ---------------------------------------------------------------------------

PHYSICIAN_TARGETS: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic payloads
# ---------------------------------------------------------------------------


class Reading(BaseModel):
    systolic: float
    diastolic: float
    pulse: Optional[float] = None
    taken_at: Optional[str] = None
    time_of_day: Optional[str] = None   # morning / afternoon / evening / night
    position: Optional[str] = None      # sitting / standing / lying


class MeasurementConditions(BaseModel):
    no_caffeine: bool = True
    no_smoking: bool = True
    no_exercise: bool = True
    bladder_empty: bool = True
    seated_5min: bool = True
    proper_posture: bool = True
    not_talking: bool = True
    cuff_bare_arm: bool = True


class PatientPayload(BaseModel):
    # Demographics
    age: float
    gender: int = 1                      # 1=M, 2=F
    is_pregnant: int = 0

    # Cardiac conditions (non-exclusive)
    has_hypertension: int = 0
    has_hfref: int = 0
    has_hfpef: int = 0
    has_hcm: int = 0
    has_dcm: int = 0
    has_cad: int = 0
    has_afib: int = 0
    has_tachycardia: int = 0
    has_bradycardia: int = 0

    # Medications (4 MVP classes)
    on_ace_or_arb: int = 0
    on_beta_blocker: int = 0
    on_loop_diuretic: int = 0
    on_nondhp_ccb: int = 0
    hours_since_bp_med: Optional[float] = None

    # Level 2 symptoms (any one → LEVEL_2)
    severe_headache_flag: int = 0
    visual_changes_flag: int = 0
    altered_mental_flag: int = 0
    chest_pain_flag: int = 0
    acute_dyspnea_flag: int = 0
    focal_neuro_flag: int = 0
    severe_epigastric_flag: int = 0
    new_headache_flag: int = 0            # pregnancy-specific
    edema_flag: int = 0                   # pregnancy-specific
    dizziness_flag: int = 0               # for bradycardia + hypotension interaction

    # Physician target override (optional)
    patient_id: Optional[str] = None

    # Measurement quality (Section 7)
    measurement_conditions: MeasurementConditions = MeasurementConditions()

    readings: List[Reading] = Field(..., min_length=1)


class ClinicalContextRequest(BaseModel):
    """Same clinical fields as PatientPayload, but no readings required."""
    age: float
    is_pregnant: int = 0
    has_hypertension: int = 0
    has_hfref: int = 0
    has_hfpef: int = 0
    has_hcm: int = 0
    has_dcm: int = 0
    has_cad: int = 0
    has_afib: int = 0
    has_tachycardia: int = 0
    has_bradycardia: int = 0
    on_ace_or_arb: int = 0
    on_beta_blocker: int = 0
    on_loop_diuretic: int = 0
    on_nondhp_ccb: int = 0
    patient_id: Optional[str] = None


class PhysicianTarget(BaseModel):
    patient_id: str
    target_sys: Optional[float] = None
    target_dia: Optional[float] = None
    lower_sys: Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/predict")
async def api_predict(payload: PatientPayload) -> Dict[str, Any]:
    try:
        data = payload.model_dump()
        data["readings"] = [r.model_dump() for r in payload.readings]
        data["measurement_conditions"] = payload.measurement_conditions.model_dump()

        target = PHYSICIAN_TARGETS.get(data.get("patient_id")) if data.get("patient_id") else None
        result = evaluate(data, physician_target=target)
        return result_to_dict(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BPException as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/clinical_context")
async def api_clinical_context(payload: ClinicalContextRequest) -> Dict[str, Any]:
    """
    Returns the threshold set that would be applied for this patient's clinical
    profile — lets physicians preview how the engine will treat this patient
    before any readings come in.
    """
    data = payload.model_dump()
    target = PHYSICIAN_TARGETS.get(data.get("patient_id")) if data.get("patient_id") else None
    thresholds = select_thresholds(data, target)
    return {
        "age_band": ("young" if data["age"] < R.AGE_BAND_MID_MIN
                     else "mid" if data["age"] < R.AGE_BAND_SENIOR_MIN
                     else "senior"),
        "applied_thresholds": {
            "low_sys": thresholds.low_sys,
            "low_dia": thresholds.low_dia,
            "level_1_high_sys": thresholds.level_1_high_sys,
            "level_1_high_dia": thresholds.level_1_high_dia,
            "level_2_sys": thresholds.level_2_sys,
            "level_2_dia": thresholds.level_2_dia,
            "cad_dia_low": thresholds.cad_dia_low,
            "source": thresholds.source,
        },
        "mandatory_provider_config_required": thresholds.mandatory_provider_config_required,
    }


@app.post("/api/physician/target")
async def api_set_physician_target(target: PhysicianTarget) -> Dict[str, Any]:
    PHYSICIAN_TARGETS[target.patient_id] = target.model_dump()
    return {"status": "ok", "patient_id": target.patient_id}


@app.get("/api/physician/target/{patient_id}")
async def api_get_physician_target(patient_id: str) -> Dict[str, Any]:
    t = PHYSICIAN_TARGETS.get(patient_id)
    if not t:
        raise HTTPException(status_code=404, detail="No target configured for this patient")
    return t


if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)
