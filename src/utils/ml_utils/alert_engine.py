"""
Rule-based alert-tier engine. Evaluates EVERY prediction request BEFORE the ML
model runs. Produces a clinically-meaningful tier (CRISIS → NORMAL → CRITICAL_LOW)
plus per-stakeholder messages (patient / caregiver / physician).

The ML prediction is a separate field in the API response — the alert tier and the
ML class coexist, with the tier providing safety guardrails and the ML providing
personalized insight.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.constants import training_pipeline as C


TIER_ORDER = [
    "CRISIS",
    "URGENCY",
    "CRITICAL_LOW",
    "LOW",
    "HIGH",
    "ELEVATED",
    "NORMAL",
]

TIER_COLORS = {
    "CRISIS": "red",
    "URGENCY": "orange",
    "CRITICAL_LOW": "black",
    "LOW": "purple",
    "HIGH": "yellow",
    "ELEVATED": "blue",
    "NORMAL": "green",
}


@dataclass
class AlertResult:
    tier: str
    contributing_factors: List[str] = field(default_factory=list)
    pulse_pressure: Optional[float] = None
    pulse_pressure_flag: bool = False


def evaluate_alert_tier(
    systolic: float,
    diastolic: float,
    pulse: Optional[float],
    symptoms: Dict[str, int],
    patient_context: Dict[str, Any],
    physician_target: Optional[Dict[str, Any]] = None,
) -> AlertResult:
    """
    Evaluate the rule-based alert tier for a single reading.

    Args:
        systolic / diastolic / pulse: the reading being judged.
        symptoms: dict of symptom flags (chest_pain_flag, severe_chest_pain_flag, etc.)
        patient_context: dict with keys like on_antihypertensive, has_diagnosed_htn, ...
        physician_target: optional per-patient target override from Dr.
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

    # Use physician target overrides if available.
    high_sys = float(physician_target["target_sys"]) if physician_target and "target_sys" in physician_target else C.HYPER_SYS_ABS
    high_dia = float(physician_target["target_dia"]) if physician_target and "target_dia" in physician_target else C.HYPER_DIA_ABS
    if is_high_risk and not physician_target:
        high_sys = C.HIGH_RISK_HYPER_SYS_ABS
        high_dia = C.HIGH_RISK_HYPER_DIA_ABS

    # --- Tier evaluation (highest severity wins) ---

    # CRISIS: >=180/120 WITH acute symptoms → call 911
    if (systolic >= C.CRISIS_SYS or diastolic >= C.CRISIS_DIA) and has_crisis_symptom:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} >= {C.CRISIS_SYS:.0f}/{C.CRISIS_DIA:.0f}")
        factors.append("Acute symptoms present")
        return AlertResult(tier="CRISIS", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)

    # URGENCY: >=180/120 WITHOUT symptoms → call physician within hours
    if systolic >= C.CRISIS_SYS or diastolic >= C.CRISIS_DIA:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} >= {C.CRISIS_SYS:.0f}/{C.CRISIS_DIA:.0f}")
        return AlertResult(tier="URGENCY", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)

    # CRITICAL_LOW: sys <70 OR dia <40, OR (sys <90 + on meds + dizziness)
    if systolic < C.CRITICAL_LOW_SYS or diastolic < C.CRITICAL_LOW_DIA:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} critically low")
        return AlertResult(tier="CRITICAL_LOW", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)
    if systolic < C.HYPO_SYS_ABS and is_on_meds and has_dizziness:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} low")
        factors.append("On antihypertensive medication")
        factors.append("Dizziness reported")
        return AlertResult(tier="CRITICAL_LOW", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)

    # LOW: sys <90 OR dia <60
    if systolic < C.HYPO_SYS_ABS or diastolic < C.HYPO_DIA_ABS:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} below normal range")
        return AlertResult(tier="LOW", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)

    # HIGH: >=140/90 (or risk-aware thresholds)
    if systolic >= high_sys or diastolic >= high_dia:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} >= {high_sys:.0f}/{high_dia:.0f}")
        if is_high_risk:
            factors.append("High-risk patient (tightened thresholds)")
        return AlertResult(tier="HIGH", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)

    # ELEVATED: stage 1 HTN range (130-139 / 80-89)
    elevated_sys = C.ELEVATED_SYS_RANGE[0] <= systolic <= C.ELEVATED_SYS_RANGE[1]
    elevated_dia = C.ELEVATED_DIA_RANGE[0] <= diastolic <= C.ELEVATED_DIA_RANGE[1]
    if elevated_sys or elevated_dia:
        factors.append(f"BP {systolic:.0f}/{diastolic:.0f} in elevated range")
        return AlertResult(tier="ELEVATED", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)

    # NORMAL
    return AlertResult(tier="NORMAL", contributing_factors=factors, pulse_pressure=pp, pulse_pressure_flag=pp_flag)


# ---------------------------------------------------------------------------
# Message templates (one per tier × stakeholder)
# ---------------------------------------------------------------------------

_MESSAGES = {
    "CRISIS": {
        "patient": (
            "Your blood pressure is dangerously high ({sys}/{dia}) and you are experiencing "
            "symptoms. Sit down and rest immediately. If symptoms do not improve within "
            "5 minutes, call 911."
        ),
        "caregiver": (
            "CRISIS ALERT — Patient BP {sys}/{dia} with acute symptoms ({symptom_list}). "
            "Monitor closely. If symptoms persist or worsen within 15 minutes, call 911. "
            "Notify their physician immediately."
        ),
        "physician": (
            "CRISIS: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}"
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
            "URGENCY ALERT — Patient BP {sys}/{dia}, no acute symptoms. "
            "Ensure patient rests and re-measures in 15 minutes. "
            "Notify their physician today."
        ),
        "physician": (
            "URGENCY: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}"
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
            "LOW BP ALERT — Patient BP {sys}/{dia}. "
            "Ensure patient is seated or lying down. Monitor for fainting. "
            "Contact their physician if symptoms persist."
        ),
        "physician": (
            "CRITICAL LOW: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{med_note}"
            "Dizziness reported: {dizziness}. "
            "Evaluate for medication adjustment."
        ),
    },
    "LOW": {
        "patient": (
            "Your blood pressure is lower than usual ({sys}/{dia}). "
            "Sit down and drink some water. If you feel dizzy or faint, contact your doctor."
        ),
        "caregiver": (
            "Low BP reading — Patient BP {sys}/{dia}. "
            "Monitor for dizziness or lightheadedness."
        ),
        "physician": (
            "LOW: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}"
            "{med_note}"
        ),
    },
    "HIGH": {
        "patient": (
            "Your blood pressure is high ({sys}/{dia}). "
            "Rest for 5 minutes and re-measure. If it stays high, contact your doctor."
        ),
        "caregiver": (
            "High BP reading — Patient BP {sys}/{dia}. "
            "Encourage rest and re-measurement."
        ),
        "physician": (
            "HIGH: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}"
            "{med_note}"
        ),
    },
    "ELEVATED": {
        "patient": (
            "Your blood pressure is slightly elevated ({sys}/{dia}). "
            "This is worth monitoring — try to relax and re-measure later today."
        ),
        "caregiver": (
            "Elevated BP reading — Patient BP {sys}/{dia}. No immediate action needed."
        ),
        "physician": (
            "ELEVATED: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}"
            "Stage 1 range. {med_note}"
        ),
    },
    "NORMAL": {
        "patient": "Your blood pressure is normal ({sys}/{dia}). Keep up the good work!",
        "caregiver": "Normal BP reading — Patient BP {sys}/{dia}.",
        "physician": (
            "NORMAL: Systolic {sys}, diastolic {dia}, pulse {pulse}. "
            "{pp_note}"
        ),
    },
}


def generate_messages(
    alert: AlertResult,
    systolic: float,
    diastolic: float,
    pulse: Optional[float],
    symptoms: Dict[str, int],
    patient_context: Dict[str, Any],
) -> Dict[str, str]:
    """Format stakeholder messages for the given alert tier."""
    active_symptoms = [k for k, v in symptoms.items() if int(v or 0) and k != "dizziness_flag"]
    symptom_list = ", ".join(s.replace("_flag", "").replace("_", " ") for s in active_symptoms) or "none"
    pp_note = f"PP {alert.pulse_pressure:.0f} mmHg{' (elevated)' if alert.pulse_pressure_flag else ''}. " if alert.pulse_pressure is not None else ""
    is_on_meds = bool(int(patient_context.get("on_antihypertensive", 0) or 0))
    med_note = "On antihypertensive medication. " if is_on_meds else ""
    has_dizziness = "Yes" if int(symptoms.get("dizziness_flag", 0) or 0) else "No"

    templates = _MESSAGES.get(alert.tier, _MESSAGES["NORMAL"])
    fmt = {
        "sys": f"{systolic:.0f}",
        "dia": f"{diastolic:.0f}",
        "pulse": f"{pulse:.0f}" if pulse else "—",
        "symptom_list": symptom_list,
        "pp_note": pp_note,
        "med_note": med_note,
        "dizziness": has_dizziness,
    }
    return {role: tmpl.format(**fmt) for role, tmpl in templates.items()}


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
    """Assemble the unified API response object."""
    messages = generate_messages(alert, systolic, diastolic, pulse, symptoms, patient_context)
    resp: Dict[str, Any] = {
        "alert_tier": alert.tier,
        "alert_tier_color": TIER_COLORS.get(alert.tier, "gray"),
        "contributing_factors": alert.contributing_factors,
        "pulse_pressure": round(alert.pulse_pressure, 1) if alert.pulse_pressure is not None else None,
        "pulse_pressure_flag": alert.pulse_pressure_flag,
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
