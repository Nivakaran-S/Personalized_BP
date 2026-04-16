"""
Feature engineering and personalized-alert target construction.
Mirrors experiment_08.ipynb so training and inference produce identical features.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.constants import training_pipeline as C


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def engineer_features(df: pd.DataFrame, dryad_stats: Dict[str, float]) -> pd.DataFrame:
    """Add the 20+ derived features the notebook builds from raw NHANES rows."""
    out = df.copy()

    sy1, sy2 = out["BPXOSY1"], out["BPXOSY2"]
    di1, di2 = out["BPXODI1"], out["BPXODI2"]
    p1, p2 = out["BPXOPLS1"], out["BPXOPLS2"]

    out["sys12mean"] = (sy1 + sy2) / 2
    out["dia12mean"] = (di1 + di2) / 2
    out["pulse12mean"] = (p1 + p2) / 2

    out["sys12std"] = pd.concat([sy1, sy2], axis=1).std(axis=1, ddof=0).fillna(0)
    out["dia12std"] = pd.concat([di1, di2], axis=1).std(axis=1, ddof=0).fillna(0)
    out["pulse12std"] = pd.concat([p1, p2], axis=1).std(axis=1, ddof=0).fillna(0)

    out["systrend21"] = sy2 - sy1
    out["diatrend21"] = di2 - di1
    out["pulsetrend21"] = p2 - p1

    out["pp12"] = out["sys12mean"] - out["dia12mean"]
    out["map12"] = (out["sys12mean"] + 2 * out["dia12mean"]) / 3

    out["syscv12"] = ((out["sys12std"] / out["sys12mean"]) * 100).replace([np.inf, -np.inf], 0).fillna(0)
    out["diacv12"] = ((out["dia12std"] / out["dia12mean"]) * 100).replace([np.inf, -np.inf], 0).fillna(0)

    gender = _col(out, "RIAGENDR")
    out["isfemale"] = (gender == 2).astype(int)

    income = _col(out, "INDFMPIR")
    out["lowincomeflag"] = (income < 1.3).fillna(False).astype(int)

    bmi = _col(out, "BMXBMI")
    out["obesitycat"] = pd.cut(
        bmi,
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=["Underweight", "Normal", "Overweight", "ObeseI", "ObeseII"],
        include_lowest=True,
    ).astype(object)

    if "antihypertensiveflag" not in out.columns:
        out["antihypertensiveflag"] = 0
    if "rxcount" not in out.columns:
        out["rxcount"] = 0
    out["antihypertensiveflag"] = out["antihypertensiveflag"].fillna(0).astype(int)
    out["rxcount"] = pd.to_numeric(out["rxcount"], errors="coerce").fillna(0)

    out = _add_clinical_flags(out)
    return out


def _binary_flag(series: pd.Series) -> pd.Series:
    """Treat 1 as Yes, 0/NaN as No (the preferred default for training + inference)."""
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1).astype(int)


def _add_clinical_flags(out: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clean binary flags from raw NHANES questionnaire codes (already 0/1/NaN
    after ingestion's `_aggregate_questionnaire`) and compute rollup indices used
    by the risk-aware target rule.
    """
    code_to_flag = {
        "BPQ020":  "has_diagnosed_htn",
        "BPQ050A": "on_antihypertensive",
        "BPQ080":  "has_high_cholesterol",
        "CDQ001":  "chest_pain_flag",
        "CDQ008":  "severe_chest_pain_flag",
        "CDQ010":  "sob_on_exertion_flag",
        "MCQ160B": "has_heart_failure",
        "MCQ160C": "has_chd",
        "MCQ160D": "has_angina",
        "MCQ160E": "has_mi",
        "MCQ160F": "has_stroke",
        "DIQ010":  "has_diabetes",
    }
    # dizziness_flag has no NHANES source code — it's an inference-only input
    # from the patient. Ensure it exists with a default of 0.
    if "dizziness_flag" not in out.columns:
        out["dizziness_flag"] = 0
    out["dizziness_flag"] = _binary_flag(out["dizziness_flag"])
    for src, dst in code_to_flag.items():
        if src in out.columns:
            out[dst] = _binary_flag(out[src])
        else:
            out[dst] = 0

    out["cardiac_history_count"] = (
        out["has_mi"] + out["has_stroke"] + out["has_heart_failure"]
        + out["has_chd"] + out["has_angina"]
    ).astype(int)
    out["acute_symptom_count"] = (
        out["chest_pain_flag"] + out["sob_on_exertion_flag"] + out["severe_chest_pain_flag"]
    ).astype(int)
    out["high_risk_profile"] = (
        (out["has_diagnosed_htn"] == 1)
        | (out["on_antihypertensive"] == 1)
        | (out["cardiac_history_count"] > 0)
        | (out["has_diabetes"] == 1)
    ).astype(int)

    # Keep antihypertensiveflag aligned with the more specific BPQ050A answer
    # (ingestion already prefers it, but re-sync here after the binary-flag step).
    out["antihypertensiveflag"] = out[["antihypertensiveflag", "on_antihypertensive"]].max(axis=1).astype(int)

    return out


def add_normalization_and_proxy_features(
    df: pd.DataFrame,
    norm_stats: Dict[str, float],
    dryad_stats: Dict[str, float],
) -> pd.DataFrame:
    """Append agez/bmiz/sys12stdz + Dryad-informed proxy indices. Requires engineer_features first."""
    out = df.copy()

    age = _col(out, "RIDAGEYR")
    bmi = _col(out, "BMXBMI")

    agez = ((age.fillna(norm_stats["age_med"]) - norm_stats["age_med"]) / (norm_stats["age_sd"] + 1e-9)).clip(-3, 3)
    bmiz = ((bmi.fillna(norm_stats["bmi_med"]) - norm_stats["bmi_med"]) / (norm_stats["bmi_sd"] + 1e-9)).clip(-3, 3)
    sys12stdz = (
        (out["sys12std"].fillna(norm_stats["std_med"]) - norm_stats["std_med"])
        / (norm_stats["std_sd"] + 1e-9)
    ).clip(-3, 3)

    out["morningsurgeproxy"] = (
        dryad_stats.get("meanmorningeveningdiff", 0.0)
        + 0.40 * agez
        + 0.30 * bmiz
        + 0.30 * sys12stdz
    )

    reth = _col(out, "RIDRETH3")
    out["nondipperrisk"] = (
        0.35 * (out["sys12mean"] >= C.HYPER_SYS_REL).astype(int)
        + 0.25 * (out["dia12mean"] >= C.HYPER_DIA_REL).astype(int)
        + 0.20 * (reth == 4).astype(int)
        + 0.10 * agez
        + 0.10 * (1 - out["antihypertensiveflag"].astype(int))
    )

    out["circadiandysregulationindex"] = (
        0.5 * out["syscv12"] + 0.3 * out["diacv12"] + 0.2 * out["nondipperrisk"]
    )

    return out


def compute_sys_floor(dryad_stats: Dict[str, float]) -> float:
    """Data-dependent systolic z-score floor (matches the notebook)."""
    mean_sys_std = float(dryad_stats.get("meansysstd", 0.0) or 0.0)
    return max(C.SYS_FLOOR_BASE, mean_sys_std * C.SYS_FLOOR_DRYAD_COEF)


def compute_normalization_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Medians/SDs used for agez/bmiz/sys12stdz. Fit on training data only."""
    age = df["RIDAGEYR"]
    bmi = df["BMXBMI"]
    sys_std = df["sys12std"]
    return {
        "age_med": float(age.median()),
        "age_sd": float(age.std()),
        "bmi_med": float(bmi.median()),
        "bmi_sd": float(bmi.std()),
        "std_med": float(sys_std.median()),
        "std_sd": float(sys_std.std()),
    }


def _risk_aware_thresholds(is_high_risk: bool) -> Dict[str, float]:
    """Pick the threshold set for label construction based on patient risk profile."""
    if is_high_risk:
        return {
            "hypo_sys_abs": C.HIGH_RISK_HYPO_SYS_ABS,
            "hypo_dia_abs": C.HIGH_RISK_HYPO_DIA_ABS,
            "hypo_sys_rel": C.HIGH_RISK_HYPO_SYS_REL,
            "hypo_dia_rel": C.HIGH_RISK_HYPO_DIA_REL,
            "hyper_sys_abs": C.HIGH_RISK_HYPER_SYS_ABS,
            "hyper_dia_abs": C.HIGH_RISK_HYPER_DIA_ABS,
            "hyper_sys_rel": C.HIGH_RISK_HYPER_SYS_REL,
            "hyper_dia_rel": C.HIGH_RISK_HYPER_DIA_REL,
        }
    return {
        "hypo_sys_abs": C.HYPO_SYS_ABS,
        "hypo_dia_abs": C.HYPO_DIA_ABS,
        "hypo_sys_rel": C.HYPO_SYS_REL,
        "hypo_dia_rel": C.HYPO_DIA_REL,
        "hyper_sys_abs": C.HYPER_SYS_ABS,
        "hyper_dia_abs": C.HYPER_DIA_ABS,
        "hyper_sys_rel": C.HYPER_SYS_REL,
        "hyper_dia_rel": C.HYPER_DIA_REL,
    }


def make_personalized_alert_type(
    row: pd.Series,
    sys_floor: float = C.SYS_FLOOR_BASE,
    dia_floor: float = C.DIA_FLOOR,
) -> Optional[str]:
    """
    Classify reading 3 (BPXOSY3/BPXODI3) relative to the patient's own baseline
    (readings 1-2 mean/std), with risk-aware thresholds.

    For patients flagged `high_risk_profile` (diagnosed HTN, on BP meds, cardiac history,
    or diabetes), tighter ACC/AHA cutoffs apply — a 135/85 spike becomes Hypertensive
    instead of Normal. For others, the default cutoffs (notebook defaults) remain.

    Additionally, a "Normal" label is bumped to "Hypertensive" when the patient reports
    acute chest pain and the reading is above the SYMPTOM_BUMP cutoffs.
    """
    try:
        sy3 = float(row["BPXOSY3"])
        di3 = float(row["BPXODI3"])
        sys_mean = float(row["sys12mean"])
        dia_mean = float(row["dia12mean"])
        sys_scale = max(float(row["sys12std"]), float(sys_floor))
        dia_scale = max(float(row["dia12std"]), float(dia_floor))
    except (KeyError, TypeError, ValueError):
        return None
    if any(pd.isna(v) for v in [sy3, di3, sys_mean, dia_mean, sys_scale, dia_scale]):
        return None

    z_sys = (sy3 - sys_mean) / (sys_scale + 1e-9)
    z_dia = (di3 - dia_mean) / (dia_scale + 1e-9)

    is_high_risk = bool(int(row.get("high_risk_profile", 0) or 0))
    t = _risk_aware_thresholds(is_high_risk)

    abs_hypo = (sy3 < t["hypo_sys_abs"]) or (di3 < t["hypo_dia_abs"])
    rel_hypo = ((sy3 < t["hypo_sys_rel"]) or (di3 < t["hypo_dia_rel"])) and (
        (z_sys <= C.Z_LOW) or (z_dia <= C.Z_LOW)
    )
    if abs_hypo or rel_hypo:
        return "Hypotensive"

    abs_hyper = (sy3 >= t["hyper_sys_abs"]) or (di3 >= t["hyper_dia_abs"])
    rel_hyper = ((sy3 >= t["hyper_sys_rel"]) or (di3 >= t["hyper_dia_rel"])) and (
        (z_sys >= C.Z_HIGH) or (z_dia >= C.Z_HIGH)
    )
    if abs_hyper or rel_hyper:
        return "Hypertensive"

    # Symptom bump: a "Normal" reading in someone reporting acute chest pain and
    # at/above the bump cutoffs escalates to Hypertensive.
    has_chest_pain = bool(int(row.get("chest_pain_flag", 0) or 0)) or bool(
        int(row.get("severe_chest_pain_flag", 0) or 0)
    )
    if has_chest_pain and (
        sy3 >= C.SYMPTOM_BUMP_SYS_CUTOFF or di3 >= C.SYMPTOM_BUMP_DIA_CUTOFF
    ):
        return "Hypertensive"

    return "Normal"


def second_reading_rules_bp(sbp: float, dbp: float) -> str:
    """Simple rule baseline from the notebook (absolute thresholds only)."""
    if sbp < C.HYPO_SYS_ABS or dbp < C.HYPO_DIA_ABS:
        return "Hypotensive"
    if sbp >= C.HYPER_SYS_ABS or dbp >= C.HYPER_DIA_ABS:
        return "Hypertensive"
    return "Normal"


def build_patient_features(payload: Dict[str, Any], feature_metadata: Dict[str, Any]) -> pd.DataFrame:
    """Convert an inference payload into the single-row DataFrame the preprocessor expects."""
    readings = payload.get("readings") or []
    if len(readings) < 3:
        raise ValueError("At least 3 BP readings are required for supervised prediction.")

    # Server-side BMI fallback: if client didn't supply BMI but gave weight + height, compute it.
    bmi = payload.get("bmi")
    weight = payload.get("weight")
    height = payload.get("height")
    if (bmi is None or (isinstance(bmi, float) and np.isnan(bmi))) and weight and height:
        try:
            bmi = float(weight) / ((float(height) / 100.0) ** 2)
        except (TypeError, ValueError, ZeroDivisionError):
            bmi = None

    # Clinical fields: prefer the newer keys but accept legacy aliases.
    on_antihyp = payload.get("on_antihypertensive")
    if on_antihyp is None:
        on_antihyp = payload.get("antihypertensive_flag", 0)

    row = {
        "BPXOSY1": readings[0].get("systolic"),
        "BPXOSY2": readings[1].get("systolic"),
        "BPXOSY3": readings[2].get("systolic"),
        "BPXODI1": readings[0].get("diastolic"),
        "BPXODI2": readings[1].get("diastolic"),
        "BPXODI3": readings[2].get("diastolic"),
        "BPXOPLS1": readings[0].get("pulse"),
        "BPXOPLS2": readings[1].get("pulse"),
        "BPXOPLS3": readings[2].get("pulse"),
        "RIDAGEYR": payload.get("age"),
        "RIAGENDR": payload.get("gender"),
        "RIDRETH3": payload.get("race_ethnicity"),
        "INDFMPIR": payload.get("income_ratio"),
        "DMDEDUC2": payload.get("education"),
        "BMXBMI": bmi,
        "BMXWT": weight,
        "BMXHT": height,
        "antihypertensiveflag": int(bool(on_antihyp)),
        "rxcount": payload.get("rx_count", 0),
        # Clinical history + symptoms — treat as 1/0 at ingestion/inference parity
        "BPQ020":  payload.get("has_diagnosed_htn", 0),
        "BPQ050A": on_antihyp,
        "BPQ080":  payload.get("has_high_cholesterol", 0),
        "CDQ001":  payload.get("chest_pain_flag", 0),
        "CDQ008":  payload.get("severe_chest_pain_flag", 0),
        "CDQ010":  payload.get("sob_on_exertion_flag", 0),
        "MCQ160B": payload.get("has_heart_failure", 0),
        "MCQ160C": payload.get("has_chd", 0),
        "MCQ160D": payload.get("has_angina", 0),
        "MCQ160E": payload.get("has_mi", 0),
        "MCQ160F": payload.get("has_stroke", 0),
        "DIQ010":  payload.get("has_diabetes", 0),
        "dizziness_flag": payload.get("dizziness_flag", 0),
    }
    df = pd.DataFrame([row])
    df = engineer_features(df, feature_metadata.get("dryad_stats", C.DRYAD_DEFAULTS))
    df = add_normalization_and_proxy_features(
        df,
        feature_metadata["norm_stats"],
        feature_metadata.get("dryad_stats", C.DRYAD_DEFAULTS),
    )
    feature_cols: List[str] = feature_metadata["numeric_features"] + feature_metadata["categorical_features"]
    return df[feature_cols]


def readings_to_series(readings: List[Dict[str, float]]) -> pd.DataFrame:
    """Turn a list of reading dicts into a DataFrame for anomaly-based inference."""
    return pd.DataFrame(
        [
            {
                "systolic": r.get("systolic"),
                "diastolic": r.get("diastolic"),
                "pulse": r.get("pulse"),
            }
            for r in readings
        ]
    )


def predict_patient_alert_supervised(
    payload: Dict[str, Any],
    supervised_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    meta = supervised_bundle["feature_metadata"]
    X = build_patient_features(payload, meta)
    preprocessor = supervised_bundle["preprocessor"]
    model = supervised_bundle["model"]
    X_t = preprocessor.transform(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba_row = model.predict_proba(X_t)[0]
        proba = {cls: float(p) for cls, p in zip(meta["class_order"], proba_row)}
    pred_idx = int(model.predict(X_t)[0])
    predicted_class = meta["class_order"][pred_idx] if isinstance(pred_idx, int) else str(pred_idx)
    return {"predicted_class": predicted_class, "probabilities": proba}


_PERSONAL_ISO_DEMO_KEYS = [
    "age", "gender", "bmi", "weight", "height",
    "race_ethnicity", "education", "income_ratio",
    "on_antihypertensive", "antihypertensive_flag", "rx_count",
    "has_diagnosed_htn", "has_high_cholesterol",
    "has_mi", "has_stroke", "has_heart_failure", "has_chd", "has_angina",
    "has_diabetes",
    "chest_pain_flag", "severe_chest_pain_flag", "sob_on_exertion_flag",
    "dizziness_flag",
]


def _build_personal_feature_row(
    systolic: float,
    diastolic: float,
    pulse: float,
    sys_mean_prior: float,
    dia_mean_prior: float,
    sys_std_prior: float,
    dia_std_prior: float,
    sys_floor: float,
    dia_floor: float,
    demographic_context: Dict[str, Any],
) -> List[float]:
    """
    Feature vector for ONE reading seen in the patient's own context.
    Includes the reading itself (varies per row), patient-specific baseline stats,
    z-score deviations from baseline, and the same demographic / clinical fields
    the supervised model consumes. Demographics are constant across rows within a
    patient — they're included so the per-patient IsoForest has the same signal
    surface as the supervised model, even though within-patient variance only
    comes from the BP-derived columns.
    """
    sys_scale = max(sys_std_prior, sys_floor)
    dia_scale = max(dia_std_prior, dia_floor)
    z_sys = (float(systolic) - sys_mean_prior) / (sys_scale + 1e-9)
    z_dia = (float(diastolic) - dia_mean_prior) / (dia_scale + 1e-9)
    dev_sys = float(systolic) - sys_mean_prior
    dev_dia = float(diastolic) - dia_mean_prior

    demo = [float(demographic_context.get(k, 0) or 0) for k in _PERSONAL_ISO_DEMO_KEYS]
    return [
        float(systolic), float(diastolic), float(pulse if pulse is not None else 0.0),
        sys_mean_prior, dia_mean_prior, sys_std_prior, dia_std_prior,
        z_sys, z_dia, dev_sys, dev_dia,
    ] + demo


def _fit_personal_iso_on_full_features(
    payload: Dict[str, Any],
    readings: list,
    meta: Dict[str, Any],
    preprocessor,  # kept for signature compatibility; unused in the new scheme
):
    """
    Fit an IsolationForest on ONE patient's own reading history.

    Each training row represents one historical reading seen against the patient's
    running baseline (mean/std of ALL prior readings up to that point), plus the
    demographic + clinical context. At test time the latest reading is scored
    against a baseline built from every prior reading.

    Returns (iso, feature_mask, test_row) — feature_mask selects features with
    non-zero variance across training rows (demographics drop out naturally; BP
    values, z-scores and deviations stay), and test_row is the feature vector
    for the latest reading already built against the full-history baseline.
    """
    demo = {k: payload.get(k, 0) for k in _PERSONAL_ISO_DEMO_KEYS}
    sys_floor = float(meta.get("sys_floor", C.SYS_FLOOR_BASE))
    dia_floor = float(meta.get("dia_floor", C.DIA_FLOOR))

    readings_df = readings_to_series(readings).astype(float)
    readings_df = readings_df.dropna(subset=["systolic", "diastolic"]).reset_index(drop=True)
    if len(readings_df) < 5:
        raise ValueError(
            "Need at least 5 valid readings (4 history + 1 latest) for the per-patient IsolationForest."
        )

    training_rows: List[List[float]] = []
    # Build one training row per historical reading, using readings[:i] as the baseline.
    # Start at i=2 so the running mean/std is meaningful.
    for i in range(2, len(readings_df) - 1):
        prior = readings_df.iloc[:i]
        sys_mean = float(prior["systolic"].mean())
        dia_mean = float(prior["diastolic"].mean())
        sys_std = float(prior["systolic"].std(ddof=0))
        dia_std = float(prior["diastolic"].std(ddof=0))
        row_i = readings_df.iloc[i]
        training_rows.append(_build_personal_feature_row(
            systolic=row_i["systolic"], diastolic=row_i["diastolic"],
            pulse=row_i["pulse"] if "pulse" in row_i else 0.0,
            sys_mean_prior=sys_mean, dia_mean_prior=dia_mean,
            sys_std_prior=sys_std, dia_std_prior=dia_std,
            sys_floor=sys_floor, dia_floor=dia_floor,
            demographic_context=demo,
        ))
    if len(training_rows) < 3:
        raise ValueError(f"Only {len(training_rows)} usable training rows; need ≥3.")

    X_t = np.asarray(training_rows, dtype=float)

    # Variance-filter: drop constant columns so IsoForest's random splits actually separate.
    variance = X_t.var(axis=0)
    feature_mask = variance > 1e-9
    if feature_mask.sum() < 2:
        feature_mask = np.ones_like(feature_mask, dtype=bool)

    # Build the test row against the FULL prior history (all but the latest reading).
    prior = readings_df.iloc[:-1]
    sys_mean = float(prior["systolic"].mean())
    dia_mean = float(prior["diastolic"].mean())
    sys_std = float(prior["systolic"].std(ddof=0))
    dia_std = float(prior["diastolic"].std(ddof=0))
    latest = readings_df.iloc[-1]
    test_row = np.asarray([_build_personal_feature_row(
        systolic=latest["systolic"], diastolic=latest["diastolic"],
        pulse=latest["pulse"] if "pulse" in latest else 0.0,
        sys_mean_prior=sys_mean, dia_mean_prior=dia_mean,
        sys_std_prior=sys_std, dia_std_prior=dia_std,
        sys_floor=sys_floor, dia_floor=dia_floor,
        demographic_context=demo,
    )], dtype=float)

    iso = IsolationForest(
        n_estimators=300,
        contamination=0.25,      # more sensitive — small personal histories have tight
        random_state=C.RANDOM_STATE,  # training distributions, so the default 0.15 misses spikes
        n_jobs=1,
    )
    iso.fit(X_t[:, feature_mask])
    return iso, feature_mask, test_row


def predict_patient_alert_unsupervised(
    payload: Dict[str, Any],
    supervised_bundle: Dict[str, Any],
    unsupervised_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For patients with a long BP history, fit an IsolationForest on-the-fly against
    this patient's own readings (personal anomaly detection), then use that ML
    decision to classify the latest reading:

      - iso.predict(latest) == +1 (inlier)   → Normal
      - iso.predict(latest) == -1 (anomaly)  → Hypertensive if latest is above the
                                                patient's baseline, else Hypotensive

    The z-score rule still runs as a cross-check and its output is returned in
    details.rule_cross_check so callers can see when the two signals disagree.
    The population IsolationForest in `unsupervised_bundle` is also queried for a
    secondary `population_anomaly_score`.
    """
    readings = payload.get("readings") or []
    if len(readings) < 5:
        raise ValueError(
            "Unsupervised prediction requires at least 5 readings "
            "(need sliding 3-reading windows over the patient's history)."
        )

    readings_df = readings_to_series(readings).astype(float)
    history_full = readings_df.iloc[:-1]
    history = history_full.dropna(subset=["systolic", "diastolic"])
    if len(history) < 4:
        raise ValueError("Need at least 4 valid prior readings for per-patient IsolationForest.")
    latest = readings_df.iloc[-1]

    patient_sys_mean = float(history["systolic"].mean())
    patient_dia_mean = float(history["diastolic"].mean())
    patient_sys_std = float(history["systolic"].std(ddof=0))
    patient_dia_std = float(history["diastolic"].std(ddof=0))

    meta = supervised_bundle["feature_metadata"]
    preprocessor = supervised_bundle["preprocessor"]
    sys_floor = float(meta.get("sys_floor", C.SYS_FLOOR_BASE))
    dia_floor = float(meta.get("dia_floor", C.DIA_FLOOR))
    sys_scale = max(patient_sys_std, sys_floor)
    dia_scale = max(patient_dia_std, dia_floor)

    # --- Per-patient IsolationForest on a reading-level feature vector that
    # includes the patient's raw BP readings, baseline stats, z-deviations, AND the
    # demographic / clinical fields the supervised model uses. Demographics are
    # constant within a patient, so the variance-filter drops them before fitting —
    # but they're still part of the input schema (user intent preserved).
    personal_iso, feature_mask, test_row = _fit_personal_iso_on_full_features(
        payload, readings, meta, preprocessor
    )
    X_test_masked = test_row[:, feature_mask]
    iso_pred = int(personal_iso.predict(X_test_masked)[0])
    anomaly_score = float(personal_iso.score_samples(X_test_masked)[0])
    threshold_score = float(personal_iso.offset_)
    n_iso_features = int(feature_mask.sum())

    # Population IsoForest score needs the preprocessor-transformed supervised-style row.
    X_test_super_t = None
    try:
        super_payload = {**payload, "readings": [readings[0], readings[1], readings[-1]]}
        X_super = build_patient_features(super_payload, meta)
        X_test_super_t = preprocessor.transform(X_super)
        if hasattr(X_test_super_t, "toarray"):
            X_test_super_t = X_test_super_t.toarray()
    except Exception:  # noqa: BLE001
        X_test_super_t = None

    sy = float(latest["systolic"])
    di = float(latest["diastolic"])
    z_sys = (sy - patient_sys_mean) / (sys_scale + 1e-9)
    z_dia = (di - patient_dia_mean) / (dia_scale + 1e-9)

    # Rule cross-check (observability only — not authoritative). If the patient's
    # personal history genuinely hovers around high BP, then a high latest reading
    # is normal FOR THEM — the ML decides, the rule just records what a population
    # rule would have said so callers can see the difference.
    row = pd.Series({
        "BPXOSY3": sy,
        "BPXODI3": di,
        "sys12mean": patient_sys_mean,
        "dia12mean": patient_dia_mean,
        "sys12std": patient_sys_std,
        "dia12std": patient_dia_std,
        "high_risk_profile": payload.get("has_diagnosed_htn", 0) or payload.get("on_antihypertensive", 0)
            or payload.get("has_mi", 0) or payload.get("has_stroke", 0) or payload.get("has_diabetes", 0),
        "chest_pain_flag": payload.get("chest_pain_flag", 0),
        "severe_chest_pain_flag": payload.get("severe_chest_pain_flag", 0),
    })
    rule_class = make_personalized_alert_type(row, sys_floor=sys_floor, dia_floor=dia_floor) or "Normal"

    # Pure ML decision: anomalous → classify by direction vs patient's own baseline,
    # otherwise Normal. No absolute thresholds, no rule fallback. "Normal for this
    # patient" is whatever the IsolationForest says is in-distribution for them.
    if iso_pred == -1:
        if z_sys > 0 or z_dia > 0:
            predicted_class = "Hypertensive"
        else:
            predicted_class = "Hypotensive"
    else:
        predicted_class = "Normal"
    decision_source = "per_patient_isolation_forest"

    # --- Population IsolationForest (from training) for observability ---
    population_anomaly_score = None
    try:
        pop_iso = unsupervised_bundle.get("isolation_forest") if unsupervised_bundle else None
        if pop_iso is not None and X_test_super_t is not None:
            population_anomaly_score = float(pop_iso.score_samples(X_test_super_t)[0])
    except Exception:  # noqa: BLE001
        population_anomaly_score = None

    return {
        "predicted_class": predicted_class,
        "probabilities": None,
        "details": {
            "decision_source": decision_source,
            "isolation_forest_label": "anomaly" if iso_pred == -1 else "inlier",
            "anomaly_score": anomaly_score,
            "anomaly_threshold": threshold_score,
            "rule_cross_check": rule_class,
            "ml_rule_agreement": predicted_class == rule_class,
            "patient_sys_mean": patient_sys_mean,
            "patient_dia_mean": patient_dia_mean,
            "patient_sys_std": patient_sys_std,
            "patient_dia_std": patient_dia_std,
            "z_sys_latest": z_sys,
            "z_dia_latest": z_dia,
            "n_history": int(len(history)),
            "n_iso_features": n_iso_features,
            "population_anomaly_score": population_anomaly_score,
        },
    }


def predict_patient_alert(
    payload: Dict[str, Any],
    supervised_bundle: Optional[Dict[str, Any]],
    unsupervised_bundle: Optional[Dict[str, Any]],
    min_for_personalization: int = C.MIN_READINGS_FOR_PERSONALIZATION,
    unsupervised_threshold: int = C.UNSUPERVISED_READINGS_THRESHOLD,
) -> Optional[Dict[str, Any]]:
    """
    Three-tier routing:
      < min_for_personalization  → None (rule-only, ML skipped)
      min..unsupervised_threshold → supervised ML
      > unsupervised_threshold    → per-patient IsolationForest
    Returns None when readings are too few for personalization (caller uses rule tier only).
    """
    readings = payload.get("readings") or []
    n = len(readings)

    if n < min_for_personalization:
        return None  # rule-only; caller shows "collecting baseline"

    if n > unsupervised_threshold and unsupervised_bundle is not None and supervised_bundle is not None:
        result = predict_patient_alert_unsupervised(payload, supervised_bundle, unsupervised_bundle)
        result["model_used"] = "unsupervised"
    else:
        if supervised_bundle is None:
            return None
        result = predict_patient_alert_supervised(payload, supervised_bundle)
        result["model_used"] = "supervised"
        result.setdefault("details", {})
    result["n_readings"] = n
    return result
