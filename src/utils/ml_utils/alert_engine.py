"""
Rule-based alert-tier engine. Evaluates EVERY prediction request BEFORE the ML
model runs. Produces a clinically-meaningful tier (CRISIS -> NORMAL -> CRITICAL_LOW),
pattern-based flags (morning surge, non-dipping, orthostatic hypotension), and
per-stakeholder messages (patient / caregiver / physician).

The ML prediction is a separate field in the API response -- the alert tier and the
ML class coexist, with the tier providing safety guardrails and the ML providing
personalized insight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean as _mean
from typing import Any, Dict, List, Optional

from src.constants import training_pipeline as C


TIER_ORDER = [
    "CRISIS", "URGENCY", "CRITICAL_LOW", "LOW", "HIGH", "ELEVATED", "NORMAL",
]

TIER_COLORS = {
    "CRISIS": "red", "URGENCY": "orange", "CRITICAL_LOW": "black",
    "LOW": "purple", "HIGH": "yellow", "ELEVATED": "blue", "NORMAL": "green",
}


@dataclass
class AlertResult:
    tier: str
    contributing_factors: List[str] = field(default_factory=list)
    pulse_pressure: Optional[float] = None
    pulse_pressure_flag: bool = False
    # Pattern flags (additive, not tier replacements)
    morning_surge_flag: bool = False
    morning_surge_delta: Optional[float] = None
    non_dipper_flag: bool = False
    non_dipper_ratio: Optional[float] = None
    orthostatic_flag: bool = False
    orthostatic_sys_drop: Optional[float] = None
    orthostatic_dia_drop: Optional[float] = None
    med_timing_note: Optional[str] = None


# ---------------------------------------------------------------------------
# Pattern-based detections (run over the full readings list)
# ---------------------------------------------------------------------------

def _detect_morning_surge(readings: List[Dict[str, Any]]) -> (bool, Optional[float]):
    morning = [r for r in readings if r.get("time_of_day") == "morning" and r.get("systolic")]
    evening_night = [r for r in readings if r.get("time_of_day") in ("evening", "night") and r.get("systolic")]
    if not morning or not evening_night:
        return False, None
    en_mean = _mean([float(r["systolic"]) for r in evening_night])
    m_max = max(float(r["systolic"]) for r in morning)
    delta = m_max - en_mean
    return delta >= C.MORNING_SURGE_SYS_THRESHOLD, round(delta, 1)


def _detect_non_dipping(readings: List[Dict[str, Any]]) -> (bool, Optional[float]):
    day = [r for r in readings if r.get("time_of_day") in ("morning", "afternoon") and r.get("systolic")]
    night = [r for r in readings if r.get("time_of_day") in ("evening", "night") and r.get("systolic")]
    if not day or not night:
        return False, None
    day_mean = _mean([float(r["systolic"]) for r in day])
    night_mean = _mean([float(r["systolic"]) for r in night])
    if day_mean == 0:
        return False, None
    ratio = round(night_mean / day_mean, 3)
    return ratio > C.NON_DIPPER_RATIO_THRESHOLD, ratio


def _detect_orthostatic(readings: List[Dict[str, Any]]) -> (bool, Optional[float], Optional[float]):
    seated = [r for r in readings if r.get("position") in ("sitting", "lying") and r.get("systolic")]
    standing = [r for r in readings if r.get("position") == "standing" and r.get("systolic")]
    if not seated or not standing:
        return False, None, None
    base_sys = _mean([float(r["systolic"]) for r in seated])
    base_dia = _mean([float(r["diastolic"]) for r in seated])
    stand_sys = _mean([float(r["systolic"]) for r in standing])
    stand_dia = _mean([float(r["diastolic"]) for r in standing])
    sys_drop = round(base_sys - stand_sys, 1)
    dia_drop = round(base_dia - stand_dia, 1)
    flag = sys_drop >= C.ORTHO_SYS_DROP or dia_drop >= C.ORTHO_DIA_DROP
    return flag, sys_drop, dia_drop


def _med_timing_note(patient_context: Dict[str, Any]) -> Optional[str]:
    is_on_meds = bool(int(patient_context.get("on_antihypertensive", 0) or 0))
    hours = patient_context.get("hours_since_bp_med")
    if not is_on_meds or hours is None:
        return None
    hours = float(hours)
    if hours <= C.MED_TIMING_PEAK_HOURS:
        return f"Reading taken during peak medication effect ({hours:.1f}h post-dose)."
    if hours >= C.MED_TIMING_TROUGH_HOURS:
        return f"Reading taken near medication trough ({hours:.1f}h post-dose). Consider whether current regimen provides adequate coverage."
    return None


# ---------------------------------------------------------------------------
# Main tier evaluation
# ---------------------------------------------------------------------------

def evaluate_alert_tier(
    systolic: float,
    diastolic: float,
    pulse: Optional[float],
    symptoms: Dict[str, int],
    patient_context: Dict[str, Any],
    readings: Optional[List[Dict[str, Any]]] = None,
    physician_target: Optional[Dict[str, Any]] = None,
) -> AlertResult:
    """
    Evaluate the rule-based alert tier for a single reading, plus pattern flags
    computed over the full readings list.
    """
    factors: List[str] = []

    pp = systolic - diastolic
    pp_flag = pp > C.PULSE_PRESSURE_HIGH
    if pp_flag:
        factors.append(f"Pulse pressure {pp:.0f} mmHg (>{C.PULSE_PRESSURE_HIGH:.0f})")

    has_crisis_symptom = any(int(symptoms.get(s, 0) or 0) for s in C.CRISIS_SYMPTOMS)
    is_on_meds = bool(int(patient_context.get("on_antihypertensive", 0) or 0))
    has_dizziness = bool(int(symptoms.get("dizziness_flag", 0) or 0))
    is_high_risk = bool(int(patient_context.get("high_risk_profile", 0) or 0))
    is_pregnant = bool(int(patient_context.get("is_pregnant", 0) or 0))

    high_sys = float(physician_target["target_sys"]) if physician_target and "target_sys" in physician_target else C.HYPER_SYS_ABS
    high_dia = float(physician_target["target_dia"]) if physician_target and "target_dia" in physician_target else C.HYPER_DIA_ABS
    if is_high_risk and not physician_target:
        high_sys = C.HIGH_RISK_HYPER_SYS_ABS
        high_dia = C.HIGH_RISK_HYPER_DIA_ABS

    # Pregnancy: ensure HIGH fires at 140/90 regardless of risk profile.
    if is_pregnant:
        high_sys = min(high_sys, C.PREGNANCY_HIGH_SYS)
        high_dia = min(high_dia, C.PREGNANCY_HIGH_DIA)

    # --- Pattern flags (from the full readings list) ---
    rdgs = readings or []
    ms_flag, ms_delta = _detect_morning_surge(rdgs)
    nd_flag, nd_ratio = _detect_non_dipping(rdgs)
    ort_flag, ort_sys, ort_dia = _detect_orthostatic(rdgs)
    mt_note = _med_timing_note(patient_context)

    if ms_flag:
        factors.append(f"Morning surge detected (+{ms_delta:.0f} mmHg morning vs evening)")
    if nd_flag:
        factors.append(f"Non-dipping pattern (night/day ratio {nd_ratio:.2f})")
    if ort_flag:
        factors.append(f"Orthostatic hypotension (drop {ort_sys:.0f}/{ort_dia:.0f} on standing)")
    if mt_note:
        factors.append(mt_note)
    if is_pregnant:
        factors.append("Pregnant patient -- pre-eclampsia screening indicated")

    # --- Tier evaluation (highest severity wins) ---
    if (systolic >= C.CRISIS_SYS or diastolic >= C.CRISIS_DIA) and has_crisis_symptom:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} >= {C.CRISIS_SYS:.0f}/{C.CRISIS_DIA:.0f}")
        factors.append("Acute symptoms present")
        tier = "CRISIS"
    elif systolic >= C.CRISIS_SYS or diastolic >= C.CRISIS_DIA:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} >= {C.CRISIS_SYS:.0f}/{C.CRISIS_DIA:.0f}")
        tier = "URGENCY"
    elif systolic < C.CRITICAL_LOW_SYS or diastolic < C.CRITICAL_LOW_DIA:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} critically low")
        tier = "CRITICAL_LOW"
    elif systolic < C.HYPO_SYS_ABS and is_on_meds and has_dizziness:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} low + on meds + dizziness")
        tier = "CRITICAL_LOW"
    elif systolic < C.HYPO_SYS_ABS or diastolic < C.HYPO_DIA_ABS:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} below normal range")
        tier = "LOW"
    elif systolic >= high_sys or diastolic >= high_dia:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} >= {high_sys:.0f}/{high_dia:.0f}")
        if is_high_risk:
            factors.append("High-risk patient (tightened thresholds)")
        tier = "HIGH"
    elif C.ELEVATED_SYS_RANGE[0] <= systolic <= C.ELEVATED_SYS_RANGE[1] or C.ELEVATED_DIA_RANGE[0] <= diastolic <= C.ELEVATED_DIA_RANGE[1]:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} in elevated range")
        tier = "ELEVATED"
    else:
        tier = "NORMAL"

    return AlertResult(
        tier=tier,
        contributing_factors=factors,
        pulse_pressure=pp,
        pulse_pressure_flag=pp_flag,
        morning_surge_flag=ms_flag,
        morning_surge_delta=ms_delta,
        non_dipper_flag=nd_flag,
        non_dipper_ratio=nd_ratio,
        orthostatic_flag=ort_flag,
        orthostatic_sys_drop=ort_sys,
        orthostatic_dia_drop=ort_dia,
        med_timing_note=mt_note,
    )


# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------

_MESSAGES = {
    "CRISIS": {
        "patient": (
            "Your blood pressure is dangerously high ({sys}/{dia}) and you are experiencing "
            "symptoms. Sit down and rest immediately. If symptoms do not improve within "
            "5 minutes, call 911."
        ),
        "caregiver": (
            "CRISIS ALERT -- Patient BP {sys}/{dia} with acute symptoms ({symptom_list}). "
            "Monitor closely. If symptoms persist or worsen within 15 minutes, call 911. "
            "Notify their physician immediately."
        ),
        "physician": (
            "CRISIS: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}{extra_notes}"
            "Concurrent symptoms: {symptom_list}. "
            "{med_note}"
            "Recommend immediate outreach."
        ),
    },
    "URGENCY": {
        "patient": (
            "Your blood pressure is very high ({sys}/{dia}). Sit down and rest for "
            "15 minutes, then re-measure. Contact your doctor within the next few hours."
        ),
        "caregiver": (
            "URGENCY ALERT -- Patient BP {sys}/{dia}, no acute symptoms. "
            "Ensure patient rests and re-measures in 15 minutes. "
            "Notify their physician today."
        ),
        "physician": (
            "URGENCY: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}{extra_notes}"
            "No acute symptoms reported. "
            "{med_note}"
            "Recommend outreach within 4 hours."
        ),
    },
    "CRITICAL_LOW": {
        "patient": (
            "Your blood pressure is very low ({sys}/{dia}). Sit or lie down immediately. "
            "Do not stand quickly. If you feel faint or dizzy, call your doctor now."
        ),
        "caregiver": (
            "LOW BP ALERT -- Patient BP {sys}/{dia}. "
            "Ensure patient is seated or lying down. Monitor for fainting. "
            "Contact their physician if symptoms persist."
        ),
        "physician": (
            "CRITICAL LOW: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{med_note}{extra_notes}"
            "Dizziness reported: {dizziness}. "
            "Evaluate for medication adjustment."
        ),
    },
    "LOW": {
        "patient": (
            "Your blood pressure is lower than usual ({sys}/{dia}). "
            "Sit down and drink some water. If you feel dizzy or faint, contact your doctor."
        ),
        "caregiver": "Low BP reading -- Patient BP {sys}/{dia}. Monitor for dizziness.",
        "physician": "LOW: Systolic {sys}, diastolic {dia}, pulse {pulse}. {pp_note}{extra_notes}{med_note}",
    },
    "HIGH": {
        "patient": (
            "Your blood pressure is high ({sys}/{dia}). "
            "Rest for 5 minutes and re-measure. If it stays high, contact your doctor."
        ),
        "caregiver": "High BP reading -- Patient BP {sys}/{dia}. Encourage rest and re-measurement.",
        "physician": "HIGH: Systolic {sys}, diastolic {dia}, pulse {pulse}. {pp_note}{extra_notes}{med_note}",
    },
    "ELEVATED": {
        "patient": (
            "Your blood pressure is slightly elevated ({sys}/{dia}). "
            "This is worth monitoring -- try to relax and re-measure later today."
        ),
        "caregiver": "Elevated BP reading -- Patient BP {sys}/{dia}. No immediate action needed.",
        "physician": "ELEVATED: Systolic {sys}, diastolic {dia}, pulse {pulse}. {pp_note}{extra_notes}Stage 1 range. {med_note}",
    },
    "NORMAL": {
        "patient": "Your blood pressure is normal ({sys}/{dia}). Keep up the good work!",
        "caregiver": "Normal BP reading -- Patient BP {sys}/{dia}.",
        "physician": "NORMAL: Systolic {sys}, diastolic {dia}, pulse {pulse}. {pp_note}{extra_notes}",
    },
}

# Pregnancy overrides for patient + physician messages.
_PREGNANCY_PATIENT_HIGH = (
    "Your blood pressure is elevated during pregnancy ({sys}/{dia}). "
    "This needs medical attention. Contact your OB or midwife today."
)
_PREGNANCY_PHYSICIAN_NOTE = (
    "Pregnant patient. Evaluate for pre-eclampsia. "
    "Consider urine protein and liver function. "
)


def generate_messages(
    alert: AlertResult,
    systolic: float,
    diastolic: float,
    pulse: Optional[float],
    symptoms: Dict[str, int],
    patient_context: Dict[str, Any],
) -> Dict[str, str]:
    active_symptoms = [k for k, v in symptoms.items() if int(v or 0) and k != "dizziness_flag"]
    symptom_list = ", ".join(s.replace("_flag", "").replace("_", " ") for s in active_symptoms) or "none"
    pp_note = f"PP {alert.pulse_pressure:.0f} mmHg{' (elevated)' if alert.pulse_pressure_flag else ''}. " if alert.pulse_pressure is not None else ""
    is_on_meds = bool(int(patient_context.get("on_antihypertensive", 0) or 0))
    is_pregnant = bool(int(patient_context.get("is_pregnant", 0) or 0))
    med_note = "On antihypertensive medication. " if is_on_meds else ""
    has_dizziness = "Yes" if int(symptoms.get("dizziness_flag", 0) or 0) else "No"

    extra_parts: List[str] = []
    if alert.morning_surge_flag:
        extra_parts.append(f"Morning surge +{alert.morning_surge_delta:.0f} mmHg.")
    if alert.non_dipper_flag:
        extra_parts.append(f"Non-dipping pattern (ratio {alert.non_dipper_ratio:.2f}).")
    if alert.orthostatic_flag:
        extra_parts.append(f"Orthostatic drop {alert.orthostatic_sys_drop:.0f}/{alert.orthostatic_dia_drop:.0f} mmHg.")
    if alert.med_timing_note:
        extra_parts.append(alert.med_timing_note)
    if is_pregnant:
        extra_parts.append(_PREGNANCY_PHYSICIAN_NOTE)
    extra_notes = " ".join(extra_parts) + " " if extra_parts else ""

    templates = _MESSAGES.get(alert.tier, _MESSAGES["NORMAL"])
    fmt = {
        "sys": f"{systolic:.0f}", "dia": f"{diastolic:.0f}",
        "pulse": f"{pulse:.0f}" if pulse else "--",
        "symptom_list": symptom_list, "pp_note": pp_note,
        "med_note": med_note, "dizziness": has_dizziness,
        "extra_notes": extra_notes,
    }
    msgs = {role: tmpl.format(**fmt) for role, tmpl in templates.items()}

    # Pregnancy patient-message override for HIGH tier.
    if is_pregnant and alert.tier == "HIGH":
        msgs["patient"] = _PREGNANCY_PATIENT_HIGH.format(**fmt)

    # Patient-facing additive warnings.
    addenda: List[str] = []
    if alert.morning_surge_flag:
        addenda.append("Your morning blood pressure is significantly higher than your evening readings. This pattern is associated with increased cardiovascular risk. Discuss with your doctor.")
    if alert.orthostatic_flag:
        addenda.append("Your blood pressure drops significantly when you stand. Move slowly when getting up. Tell your doctor -- medication adjustment may be needed.")
    if addenda:
        msgs["patient"] += " " + " ".join(addenda)

    return msgs


def build_full_response(
    alert: AlertResult,
    ml_prediction: Optional[Dict[str, Any]],
    personalization_status: str,
    n_readings: int,
    systolic: float,
    diastolic: float,
    pulse: Optional[float],
    symptoms: Dict[str, Any],
    patient_context: Dict[str, Any],
) -> Dict[str, Any]:
    messages = generate_messages(alert, systolic, diastolic, pulse, symptoms, patient_context)
    resp: Dict[str, Any] = {
        "alert_tier": alert.tier,
        "alert_tier_color": TIER_COLORS.get(alert.tier, "gray"),
        "contributing_factors": alert.contributing_factors,
        "pulse_pressure": round(alert.pulse_pressure, 1) if alert.pulse_pressure is not None else None,
        "pulse_pressure_flag": alert.pulse_pressure_flag,
        "morning_surge_flag": alert.morning_surge_flag,
        "morning_surge_delta": alert.morning_surge_delta,
        "non_dipper_flag": alert.non_dipper_flag,
        "non_dipper_ratio": alert.non_dipper_ratio,
        "orthostatic_flag": alert.orthostatic_flag,
        "orthostatic_sys_drop": alert.orthostatic_sys_drop,
        "orthostatic_dia_drop": alert.orthostatic_dia_drop,
        "med_timing_note": alert.med_timing_note,
        "messages": messages,
        "personalization_status": personalization_status,
        "n_readings": n_readings,
    }
    if personalization_status == "collecting_baseline":
        remaining = max(0, C.MIN_READINGS_FOR_PERSONALIZATION - n_readings)
        resp["personalization_message"] = (
            f"Personalized alerts begin after {C.MIN_READINGS_FOR_PERSONALIZATION} readings. "
            f"{remaining} more reading{'s' if remaining != 1 else ''} needed. "
            "Current classification uses standard AHA thresholds."
        )
    if ml_prediction is not None:
        resp["ml_prediction"] = ml_prediction
    return resp
