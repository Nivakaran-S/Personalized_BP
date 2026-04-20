"""
Rule-based clinical alert engine for HealPlace Cardio.

Implements Dr. Manisha Singal's signed-off specification:
- 2025 AHA/ACC tiers (Normal / Elevated / Stage 1 / Stage 2 / Severe Stage 2 / Emergency)
- Age-band modifiers (18-39 / 40-64 / 65+)
- Cardiac condition overrides (HFrEF, HFpEF, HCM, DCM, CAD, AFib, Tachy, Brady, HTN)
- Pregnancy-specific thresholds (Section 4)
- Level 2 symptom overrides (Section 2.3)
- Medication-class interactions (ACE/ARB, beta-blocker, loop diuretic, non-DHP CCB)
- Reading averaging (2-3 per session)
- Pre-measurement quality flagging
- Pattern flags: pulse pressure, morning surge, non-dipping, orthostatic
- Provider-target override with ">= upper_target + 20" heuristic
- Three-stakeholder messaging (patient / caregiver / physician)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean as _mean
from typing import Any, Dict, List, Optional, Tuple

from src.constants import alert_rules as R


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def age_band(age: float) -> str:
    if age < R.AGE_BAND_MID_MIN:
        return "young"
    if age < R.AGE_BAND_SENIOR_MIN:
        return "mid"
    return "senior"


def _i(payload: Dict[str, Any], key: str) -> int:
    """Safe int-coerce of a payload flag (treats None/missing/truthy as 0/1)."""
    v = payload.get(key, 0)
    try:
        return int(bool(int(v or 0)))
    except (TypeError, ValueError):
        return 0


def _bool_flags(payload: Dict[str, Any], names: List[str]) -> List[str]:
    return [n for n in names if _i(payload, n)]


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------


def average_latest_readings(
    readings: List[Dict[str, Any]], n: int = R.DEFAULT_AVERAGING_WINDOW
) -> Tuple[Optional[Dict[str, Optional[float]]], int]:
    """Return the mean of the last `n` valid readings, plus the count used."""
    valid = [
        r for r in readings
        if r.get("systolic") is not None and r.get("diastolic") is not None
    ]
    used = valid[-n:] if valid else []
    if not used:
        return None, 0
    sys_vals = [float(r["systolic"]) for r in used]
    dia_vals = [float(r["diastolic"]) for r in used]
    pulse_vals = [float(r["pulse"]) for r in used if r.get("pulse") is not None]
    return (
        {
            "systolic": round(_mean(sys_vals), 1),
            "diastolic": round(_mean(dia_vals), 1),
            "pulse": round(_mean(pulse_vals), 1) if pulse_vals else None,
        },
        len(used),
    )


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


@dataclass
class AppliedThresholds:
    low_sys: float
    low_dia: float
    level_1_high_sys: float
    level_1_high_dia: float
    level_2_sys: float
    level_2_dia: float
    source: str
    cad_dia_low: Optional[float] = None
    mandatory_provider_config_required: bool = False


def select_thresholds(
    payload: Dict[str, Any],
    physician_target: Optional[Dict[str, Any]] = None,
) -> AppliedThresholds:
    """Pick the tier thresholds that apply to this patient."""
    band = age_band(float(payload.get("age") or 0))
    is_pregnant = bool(_i(payload, "is_pregnant"))

    low_sys = R.LEVEL_1_LOW_SYS_SENIOR if band == "senior" else R.LEVEL_1_LOW_SYS_YOUNG_MID
    low_dia = R.LEVEL_1_LOW_DIA
    l1h_sys = R.LEVEL_1_HIGH_SYS
    l1h_dia = R.LEVEL_1_HIGH_DIA
    l2_sys = R.LEVEL_2_SYS
    l2_dia = R.LEVEL_2_DIA
    cad_dia_low: Optional[float] = None
    mandatory_cfg = False
    source_parts: List[str] = [f"age_band={band}"]

    if is_pregnant:
        l1h_sys = min(l1h_sys, R.PREGNANCY_LEVEL_1_HIGH_SYS)
        l1h_dia = min(l1h_dia, R.PREGNANCY_LEVEL_1_HIGH_DIA)
        l2_sys = min(l2_sys, R.PREGNANCY_LEVEL_2_SYS)
        l2_dia = min(l2_dia, R.PREGNANCY_LEVEL_2_DIA)
        source_parts.append("pregnancy")

    # Condition-specific lower bound. Conditions OVERRIDE the age default; if multiple
    # conditions are present, the strictest (highest) override wins. HFrEF tolerates
    # lower SBP (85 floor), HFpEF and HCM require stricter floors.
    condition_sources: List[str] = []
    condition_low: Optional[float] = None
    if _i(payload, "has_hfref") or _i(payload, "has_dcm"):
        condition_low = max(condition_low or 0.0, R.HFREF_LOWER_SYS)
        mandatory_cfg = True
        condition_sources.append("HFrEF/DCM")
    if _i(payload, "has_hfpef"):
        condition_low = max(condition_low or 0.0, R.HFPEF_LOWER_SYS)
        condition_sources.append("HFpEF")
    if _i(payload, "has_hcm"):
        condition_low = max(condition_low or 0.0, R.HCM_LOWER_SYS)
        mandatory_cfg = True
        condition_sources.append("HCM")
    if condition_low is not None:
        low_sys = condition_low
    if _i(payload, "has_cad"):
        cad_dia_low = R.CAD_LOWER_DIA
        condition_sources.append("CAD")
    if condition_sources:
        source_parts.append("conditions=" + ",".join(condition_sources))

    if physician_target:
        target_sys = float(physician_target.get("target_sys") or 0)
        target_dia = float(physician_target.get("target_dia") or 0)
        if target_sys:
            l1h_sys = target_sys + R.PROVIDER_TARGET_UPPER_HEURISTIC
            mandatory_cfg = False  # provider has configured
        if target_dia:
            l1h_dia = target_dia + R.PROVIDER_TARGET_UPPER_HEURISTIC
        if physician_target.get("lower_sys") is not None:
            low_sys = float(physician_target["lower_sys"])
        source_parts.append("physician_target")

    return AppliedThresholds(
        low_sys=low_sys,
        low_dia=low_dia,
        level_1_high_sys=l1h_sys,
        level_1_high_dia=l1h_dia,
        level_2_sys=l2_sys,
        level_2_dia=l2_dia,
        source=" | ".join(source_parts),
        cad_dia_low=cad_dia_low,
        mandatory_provider_config_required=mandatory_cfg,
    )


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def _in_range(value: float, rng: Tuple[float, float]) -> bool:
    return rng[0] <= value <= rng[1]


def assign_tier(sys_v: float, dia_v: float, thresholds: AppliedThresholds) -> str:
    """Pick the tier label based on numeric thresholds (no symptoms)."""
    if sys_v >= thresholds.level_2_sys or dia_v >= thresholds.level_2_dia:
        return "LEVEL_2"
    if sys_v >= thresholds.level_1_high_sys or dia_v >= thresholds.level_1_high_dia:
        return "LEVEL_1_HIGH"
    if sys_v < thresholds.low_sys or dia_v < thresholds.low_dia:
        return "LEVEL_1_LOW"
    if thresholds.cad_dia_low is not None and dia_v < thresholds.cad_dia_low:
        return "LEVEL_1_LOW"
    if _in_range(sys_v, R.STAGE_2_SYS_RANGE) or _in_range(dia_v, R.STAGE_2_DIA_RANGE):
        return "STAGE_2"
    if _in_range(sys_v, R.STAGE_1_SYS_RANGE) or _in_range(dia_v, R.STAGE_1_DIA_RANGE):
        return "STAGE_1"
    if _in_range(sys_v, R.ELEVATED_SYS_RANGE) and dia_v < R.NORMAL_DIA_MAX:
        return "ELEVATED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# Pattern flags
# ---------------------------------------------------------------------------


def _detect_morning_surge(
    readings: List[Dict[str, Any]],
) -> Tuple[bool, Optional[float]]:
    morning = [r for r in readings if r.get("time_of_day") == "morning" and r.get("systolic") is not None]
    evening_night = [r for r in readings if r.get("time_of_day") in ("evening", "night") and r.get("systolic") is not None]
    if not morning or not evening_night:
        return False, None
    en_mean = _mean([float(r["systolic"]) for r in evening_night])
    m_max = max(float(r["systolic"]) for r in morning)
    delta = m_max - en_mean
    return delta >= R.MORNING_SURGE_SYS_THRESHOLD, round(delta, 1)


def _detect_non_dipping(
    readings: List[Dict[str, Any]],
) -> Tuple[bool, Optional[float]]:
    day = [r for r in readings if r.get("time_of_day") in ("morning", "afternoon") and r.get("systolic") is not None]
    night = [r for r in readings if r.get("time_of_day") in ("evening", "night") and r.get("systolic") is not None]
    if not day or not night:
        return False, None
    dm = _mean([float(r["systolic"]) for r in day])
    nm = _mean([float(r["systolic"]) for r in night])
    if dm == 0:
        return False, None
    ratio = round(nm / dm, 3)
    return ratio > R.NON_DIPPER_RATIO_THRESHOLD, ratio


def _detect_orthostatic(
    readings: List[Dict[str, Any]],
) -> Tuple[bool, Optional[float], Optional[float]]:
    seated = [r for r in readings if r.get("position") in ("sitting", "lying") and r.get("systolic") is not None]
    standing = [r for r in readings if r.get("position") == "standing" and r.get("systolic") is not None]
    if not seated or not standing:
        return False, None, None
    bs = _mean([float(r["systolic"]) for r in seated])
    bd = _mean([float(r["diastolic"]) for r in seated])
    ss = _mean([float(r["systolic"]) for r in standing])
    sd = _mean([float(r["diastolic"]) for r in standing])
    sys_drop = round(bs - ss, 1)
    dia_drop = round(bd - sd, 1)
    return (sys_drop >= R.ORTHO_SYS_DROP or dia_drop >= R.ORTHO_DIA_DROP), sys_drop, dia_drop


def _med_timing_note(payload: Dict[str, Any]) -> Optional[str]:
    on_any = any(
        _i(payload, k)
        for k in ("on_ace_or_arb", "on_beta_blocker", "on_loop_diuretic", "on_nondhp_ccb")
    )
    if not on_any:
        return None
    hours = payload.get("hours_since_bp_med")
    if hours is None:
        return None
    hours = float(hours)
    if hours <= R.MED_TIMING_PEAK_HOURS:
        return f"Reading taken during peak medication effect ({hours:.1f}h post-dose)."
    if hours >= R.MED_TIMING_TROUGH_HOURS:
        return (
            f"Reading taken near medication trough ({hours:.1f}h post-dose). "
            "Consider whether current regimen provides adequate coverage."
        )
    return None


# ---------------------------------------------------------------------------
# Heart rate alerts
# ---------------------------------------------------------------------------


def evaluate_hr(
    readings: List[Dict[str, Any]],
    pulse: Optional[float],
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate heart-rate alerts (AFib / tachy / brady + beta-blocker suppression)."""
    result: Dict[str, Any] = {"hr_alert": None, "hr_factors": []}
    if pulse is None:
        return result

    has_afib = _i(payload, "has_afib")
    has_tachy = _i(payload, "has_tachycardia")
    on_bb = _i(payload, "on_beta_blocker")
    has_dizziness = _i(payload, "dizziness_flag")

    if has_afib:
        if pulse >= R.AFIB_HR_HIGH:
            result["hr_alert"] = "AFIB_HR_HIGH"
            result["hr_factors"].append(f"AFib HR {pulse:.0f} >= {R.AFIB_HR_HIGH:.0f}")
        elif pulse < R.AFIB_HR_LOW:
            result["hr_alert"] = "AFIB_HR_LOW"
            result["hr_factors"].append(f"AFib HR {pulse:.0f} < {R.AFIB_HR_LOW:.0f}")
        return result

    if has_tachy:
        recent = [
            float(r["pulse"])
            for r in readings[-R.TACHY_CONSECUTIVE_READINGS:]
            if r.get("pulse") is not None
        ]
        if len(recent) >= R.TACHY_CONSECUTIVE_READINGS and all(p > R.TACHY_HR for p in recent):
            result["hr_alert"] = "TACHYCARDIA"
            result["hr_factors"].append(
                f"HR > {R.TACHY_HR:.0f} on {R.TACHY_CONSECUTIVE_READINGS} consecutive readings"
            )

    # Bradycardia — applies regardless of has_bradycardia flag for patient safety.
    #   HR < 40  -> always alert (asymptomatic threshold per spec)
    #   HR 40-60 -> alert UNLESS on_beta_blocker AND HR >= 50 (therapeutic suppression)
    #   HR >= 60 -> no brady alert
    if pulse < R.BRADY_ASYMPTOMATIC_HR:
        result["hr_alert"] = "BRADYCARDIA"
        result["hr_factors"].append(
            f"HR {pulse:.0f} < {R.BRADY_ASYMPTOMATIC_HR:.0f} (asymptomatic threshold)"
        )
    elif pulse < R.BETA_BLOCKER_SUPPRESSION_MAX:
        in_bb_range = pulse >= R.BETA_BLOCKER_SUPPRESSION_MIN
        if on_bb and in_bb_range:
            result["hr_factors"].append(
                f"HR {pulse:.0f} within beta-blocker therapeutic range -- alert suppressed"
            )
        else:
            result["hr_alert"] = "BRADYCARDIA"
            reason = "with dizziness" if has_dizziness else "below normal range"
            result["hr_factors"].append(f"HR {pulse:.0f} < 60 {reason}")

    return result


# ---------------------------------------------------------------------------
# Medication contraindications
# ---------------------------------------------------------------------------


def evaluate_contraindications(payload: Dict[str, Any]) -> List[str]:
    alerts: List[str] = []
    if _i(payload, "is_pregnant") and _i(payload, "on_ace_or_arb"):
        alerts.append(R.CONTRAINDICATION_ACE_ARB_PREGNANCY)
    if _i(payload, "on_nondhp_ccb") and (_i(payload, "has_hfref") or _i(payload, "has_dcm")):
        alerts.append(R.CONTRAINDICATION_NONDHP_CCB_HFREF)
    return alerts


# ---------------------------------------------------------------------------
# Stakeholder messages
# ---------------------------------------------------------------------------


def _stakeholder_messages(
    tier: str,
    sys_v: float,
    dia_v: float,
    pulse: Optional[float],
    payload: Dict[str, Any],
    active_symptoms: List[str],
    contraindications: List[str],
    thresholds: AppliedThresholds,
    hr_info: Dict[str, Any],
) -> Dict[str, str]:
    pulse_str = f"{pulse:.0f}" if pulse is not None else "--"
    is_pregnant = bool(_i(payload, "is_pregnant"))
    preg_note = (
        "Evaluate for pre-eclampsia. Consider urine protein and liver function. "
        if is_pregnant else ""
    )
    ci_note = " ".join(contraindications) + " " if contraindications else ""
    hr_note = " ".join(hr_info["hr_factors"]) + " " if hr_info["hr_factors"] else ""
    sym_list = (
        ", ".join(s.replace("_flag", "").replace("_", " ") for s in active_symptoms)
        or "none"
    )

    if tier == "LEVEL_2":
        patient = (
            f"Your blood pressure is dangerously high ({sys_v:.0f}/{dia_v:.0f}). "
            "Sit down and rest immediately. If you have chest pain, severe headache, "
            "vision changes, or shortness of breath -- call 911 now."
        )
        caregiver = (
            f"LEVEL 2 ALERT -- Patient BP {sys_v:.0f}/{dia_v:.0f}. "
            f"Symptoms: {sym_list}. Monitor closely; if symptoms persist or worsen, "
            "call 911. Notify physician immediately."
        )
        physician = (
            f"LEVEL 2 (Emergency): BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. "
            f"{hr_note}{preg_note}{ci_note}Symptoms: {sym_list}. "
            "Immediate outreach recommended. Assess for acute target organ damage."
        )
    elif tier == "LEVEL_1_HIGH":
        patient = (
            f"Your blood pressure is high ({sys_v:.0f}/{dia_v:.0f}). "
            "Rest for 15 minutes, then re-measure. Contact your doctor today."
        )
        if is_pregnant:
            patient = (
                f"Your blood pressure is elevated during pregnancy ({sys_v:.0f}/{dia_v:.0f}). "
                "Contact your OB or midwife today."
            )
        caregiver = (
            f"LEVEL 1 HIGH -- Patient BP {sys_v:.0f}/{dia_v:.0f}. "
            "Ensure patient rests and re-measures. Notify physician today."
        )
        physician = (
            f"LEVEL 1 HIGH: BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. "
            f"{hr_note}{preg_note}{ci_note}Thresholds: {thresholds.source}. "
            "Recommend outreach within 4 hours."
        )
    elif tier == "LEVEL_1_LOW":
        patient = (
            f"Your blood pressure is low ({sys_v:.0f}/{dia_v:.0f}). "
            "Sit or lie down, drink some water. If you feel faint or dizzy, call your doctor."
        )
        caregiver = (
            f"LEVEL 1 LOW -- Patient BP {sys_v:.0f}/{dia_v:.0f}. "
            "Ensure patient is seated or lying down. Monitor for fainting."
        )
        physician = (
            f"LEVEL 1 LOW: BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. "
            f"{hr_note}{ci_note}Thresholds: {thresholds.source}. "
            "Evaluate for over-treatment or intravascular volume depletion."
        )
    elif tier == "STAGE_2":
        patient = (
            f"Your blood pressure is elevated ({sys_v:.0f}/{dia_v:.0f}). "
            "Keep measuring and discuss with your doctor at your next visit."
        )
        caregiver = f"Stage 2 reading -- Patient BP {sys_v:.0f}/{dia_v:.0f}. Encourage monitoring."
        physician = (
            f"STAGE 2 (dashboard): BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. "
            f"{hr_note}{ci_note}No push alert issued."
        )
    elif tier == "STAGE_1":
        patient = (
            f"Your blood pressure is slightly elevated ({sys_v:.0f}/{dia_v:.0f}). Keep monitoring."
        )
        caregiver = f"Stage 1 reading -- Patient BP {sys_v:.0f}/{dia_v:.0f}. No immediate action."
        physician = f"STAGE 1 (dashboard): BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. {hr_note}"
    elif tier == "ELEVATED":
        patient = f"Your blood pressure is at the upper end of normal ({sys_v:.0f}/{dia_v:.0f})."
        caregiver = f"Elevated reading -- Patient BP {sys_v:.0f}/{dia_v:.0f}."
        physician = f"ELEVATED (dashboard): BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. {hr_note}"
    else:  # NORMAL
        patient = f"Your blood pressure is normal ({sys_v:.0f}/{dia_v:.0f}). Keep it up!"
        caregiver = f"Normal reading -- Patient BP {sys_v:.0f}/{dia_v:.0f}."
        physician = f"NORMAL: BP {sys_v:.0f}/{dia_v:.0f}, pulse {pulse_str}. {hr_note}"

    return {
        "patient": patient.strip(),
        "caregiver": caregiver.strip(),
        "physician": physician.strip(),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@dataclass
class AlertResult:
    tier: str
    push_alert: bool
    alert_level_numeric: int
    applied_reading: Dict[str, Any]
    applied_thresholds: Dict[str, Any]
    age_band: str
    contributing_factors: List[str]
    pattern_flags: Dict[str, Any]
    contraindication_alerts: List[str]
    measurement_quality: str
    mandatory_provider_config_required: bool
    clinical_context_flag: Optional[str]
    measurement_protocol_note: Optional[str]
    hr_alert: Optional[str]
    insufficient_readings: bool
    messages: Dict[str, str]
    symptoms_active: List[str] = field(default_factory=list)


_TIER_NUMERIC = {
    "LEVEL_2": 2,
    "LEVEL_1_HIGH": 1,
    "LEVEL_1_LOW": 1,
    "STAGE_2": 0,
    "STAGE_1": 0,
    "ELEVATED": 0,
    "NORMAL": 0,
}


def evaluate(
    payload: Dict[str, Any],
    physician_target: Optional[Dict[str, Any]] = None,
) -> AlertResult:
    """
    Main entry point for the rule engine.
    `payload` is the cleaned PatientPayload dict (readings + clinical fields).
    Returns a fully-populated AlertResult.
    """
    readings: List[Dict[str, Any]] = payload.get("readings") or []

    # AFib reading gate.
    is_afib = _i(payload, "has_afib")
    insufficient = False
    protocol_note: Optional[str] = None
    if is_afib and len(readings) < R.AFIB_MIN_READINGS:
        insufficient = True
        protocol_note = (
            f"AFib patients require at least {R.AFIB_MIN_READINGS} readings per session "
            "before an alert is generated."
        )

    averaged, n_used = average_latest_readings(readings, R.DEFAULT_AVERAGING_WINDOW)
    if averaged is None:
        raise ValueError("No valid readings provided.")

    sys_v = float(averaged["systolic"])
    dia_v = float(averaged["diastolic"])
    pulse_v = averaged.get("pulse")

    thresholds = select_thresholds(payload, physician_target)
    band = age_band(float(payload.get("age") or 0))

    active_symptoms = _bool_flags(payload, R.LEVEL_2_SYMPTOMS)
    if _i(payload, "is_pregnant"):
        for s in _bool_flags(payload, R.PREGNANCY_LEVEL_2_SYMPTOMS):
            if s not in active_symptoms:
                active_symptoms.append(s)

    contraindications = evaluate_contraindications(payload)

    numeric_tier = assign_tier(sys_v, dia_v, thresholds)

    teratogenic = any("teratogenic" in c.lower() for c in contraindications)
    if active_symptoms or teratogenic:
        tier = "LEVEL_2"
    elif insufficient:
        tier = "NORMAL"
    else:
        tier = numeric_tier

    hr_info = evaluate_hr(readings, pulse_v, payload)

    ms_flag, ms_delta = _detect_morning_surge(readings)
    nd_flag, nd_ratio = _detect_non_dipping(readings)
    ort_flag, ort_sys, ort_dia = _detect_orthostatic(readings)
    pp = sys_v - dia_v
    pp_flag = pp > R.PULSE_PRESSURE_HIGH
    mt_note = _med_timing_note(payload)

    factors: List[str] = [
        f"BP {sys_v:.0f}/{dia_v:.0f} (averaged over {n_used} reading{'s' if n_used != 1 else ''})"
    ]
    if active_symptoms:
        factors.append(
            "Level 2 symptoms present: "
            + ", ".join(s.replace("_flag", "") for s in active_symptoms)
        )
    if contraindications:
        factors.extend(contraindications)
    if hr_info["hr_factors"]:
        factors.extend(hr_info["hr_factors"])
    if pp_flag:
        factors.append(f"Pulse pressure {pp:.0f} mmHg (> {R.PULSE_PRESSURE_HIGH:.0f})")
    if ms_flag and ms_delta is not None:
        factors.append(f"Morning surge +{ms_delta:.0f} mmHg")
    if nd_flag and nd_ratio is not None:
        factors.append(f"Non-dipping (night/day ratio {nd_ratio:.2f})")
    if ort_flag and ort_sys is not None and ort_dia is not None:
        factors.append(f"Orthostatic drop {ort_sys:.0f}/{ort_dia:.0f} mmHg")
    if mt_note:
        factors.append(mt_note)
    if thresholds.cad_dia_low is not None and dia_v < thresholds.cad_dia_low:
        factors.append(f"DBP {dia_v:.0f} < {thresholds.cad_dia_low:.0f} (CAD caution)")
    if _i(payload, "on_loop_diuretic") and sys_v < R.LEVEL_1_LOW_SYS_YOUNG_MID:
        factors.append("Diuretic-associated hypotension risk")
    if insufficient and protocol_note:
        factors.append(protocol_note)
    if thresholds.mandatory_provider_config_required:
        factors.append("Provider-configured target required for this condition")

    # Measurement quality (Section 7 -- flag only, retain in alert logic).
    mc = payload.get("measurement_conditions") or {}
    suboptimal = any(
        mc.get(k) is False
        for k in (
            "no_caffeine",
            "no_smoking",
            "no_exercise",
            "bladder_empty",
            "seated_5min",
            "proper_posture",
            "not_talking",
            "cuff_bare_arm",
        )
    )
    quality = "suboptimal" if suboptimal else "optimal"
    if suboptimal:
        factors.append("Measurement conditions flagged as suboptimal")

    messages = _stakeholder_messages(
        tier, sys_v, dia_v, pulse_v, payload, active_symptoms, contraindications,
        thresholds, hr_info,
    )

    return AlertResult(
        tier=tier,
        push_alert=tier in R.PUSH_ALERT_TIERS,
        alert_level_numeric=_TIER_NUMERIC.get(tier, 0),
        applied_reading={
            "systolic": sys_v,
            "diastolic": dia_v,
            "pulse": pulse_v,
            "n_averaged": n_used,
        },
        applied_thresholds={
            "low_sys": thresholds.low_sys,
            "low_dia": thresholds.low_dia,
            "level_1_high_sys": thresholds.level_1_high_sys,
            "level_1_high_dia": thresholds.level_1_high_dia,
            "level_2_sys": thresholds.level_2_sys,
            "level_2_dia": thresholds.level_2_dia,
            "cad_dia_low": thresholds.cad_dia_low,
            "source": thresholds.source,
        },
        age_band=band,
        contributing_factors=factors,
        pattern_flags={
            "pulse_pressure": round(pp, 1),
            "pulse_pressure_flag": pp_flag,
            "morning_surge_flag": ms_flag,
            "morning_surge_delta": ms_delta,
            "non_dipper_flag": nd_flag,
            "non_dipper_ratio": nd_ratio,
            "orthostatic_flag": ort_flag,
            "orthostatic_sys_drop": ort_sys,
            "orthostatic_dia_drop": ort_dia,
            "med_timing_note": mt_note,
        },
        contraindication_alerts=contraindications,
        measurement_quality=quality,
        mandatory_provider_config_required=thresholds.mandatory_provider_config_required,
        clinical_context_flag=R.AGE_BAND_CONTEXT.get(band),
        measurement_protocol_note=protocol_note,
        hr_alert=hr_info["hr_alert"],
        insufficient_readings=insufficient,
        messages=messages,
        symptoms_active=active_symptoms,
    )


def result_to_dict(result: AlertResult) -> Dict[str, Any]:
    """Shape AlertResult for JSON response."""
    return {
        "alert_tier": result.tier,
        "alert_tier_color": R.TIER_COLORS.get(result.tier, "gray"),
        "alert_level_numeric": result.alert_level_numeric,
        "push_alert": result.push_alert,
        "applied_reading": result.applied_reading,
        "applied_thresholds": result.applied_thresholds,
        "age_band": result.age_band,
        "contributing_factors": result.contributing_factors,
        "pattern_flags": result.pattern_flags,
        "contraindication_alerts": result.contraindication_alerts,
        "measurement_quality": result.measurement_quality,
        "mandatory_provider_config_required": result.mandatory_provider_config_required,
        "clinical_context_flag": result.clinical_context_flag,
        "measurement_protocol_note": result.measurement_protocol_note,
        "hr_alert": result.hr_alert,
        "insufficient_readings": result.insufficient_readings,
        "symptoms_active": result.symptoms_active,
        "messages": result.messages,
    }
