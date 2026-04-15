"""
Feature engineering and personalized-alert target construction.
Mirrors experiment_08.ipynb so training and inference produce identical features.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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


def predict_patient_alert_unsupervised(
    payload: Dict[str, Any],
    supervised_bundle: Dict[str, Any],
    unsupervised_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    For patients with a long BP history, run the saved IsolationForest against a
    feature vector built from the patient's own baseline (mean/std of all readings
    except the latest) and the latest reading. Anomalies are split into Hypo vs
    Hyper by comparing the latest reading to the patient's own baseline (not the
    population median), so labels are truly personalized.
    """
    iso = unsupervised_bundle["isolation_forest"]

    readings = payload.get("readings") or []
    if len(readings) < 3:
        raise ValueError("Unsupervised prediction requires at least 3 readings.")

    readings_df = readings_to_series(readings).astype(float)
    history = readings_df.iloc[:-1]
    latest = readings_df.iloc[-1]

    patient_sys_mean = float(history["systolic"].mean())
    patient_dia_mean = float(history["diastolic"].mean())
    patient_sys_std = float(history["systolic"].std(ddof=0))
    patient_dia_std = float(history["diastolic"].std(ddof=0))
    patient_pulse_mean = float(history["pulse"].mean()) if "pulse" in history else 0.0

    meta = supervised_bundle["feature_metadata"]
    sys_floor = float(meta.get("sys_floor", C.SYS_FLOOR_BASE))
    dia_floor = float(meta.get("dia_floor", C.DIA_FLOOR))
    sys_scale = max(patient_sys_std, sys_floor)
    dia_scale = max(patient_dia_std, dia_floor)

    # Synthetic 3-reading payload that preserves the patient's own baseline.
    synthetic = {
        **payload,
        "readings": [
            {"systolic": patient_sys_mean, "diastolic": patient_dia_mean, "pulse": patient_pulse_mean},
            {"systolic": patient_sys_mean, "diastolic": patient_dia_mean, "pulse": patient_pulse_mean},
            {"systolic": float(latest["systolic"]), "diastolic": float(latest["diastolic"]),
             "pulse": float(latest["pulse"]) if "pulse" in latest else patient_pulse_mean},
        ],
    }
    X = build_patient_features(synthetic, meta)

    # Inject the patient's real sys12std / dia12std so downstream features use the
    # long-history spread, then recompute everything that depends on them so the
    # feature vector seen by the IsolationForest stays internally consistent.
    if "sys12std" in X.columns:
        X.loc[:, "sys12std"] = patient_sys_std
    if "dia12std" in X.columns:
        X.loc[:, "dia12std"] = patient_dia_std
    if "pulse12std" in X.columns:
        patient_pulse_std = float(history["pulse"].std(ddof=0)) if "pulse" in history else 0.0
        X.loc[:, "pulse12std"] = patient_pulse_std

    def _safe_cv(std_col: str, mean_col: str) -> float:
        if std_col not in X.columns or mean_col not in X.columns:
            return 0.0
        denom = float(X[mean_col].iloc[0])
        num = float(X[std_col].iloc[0])
        return (num / denom * 100.0) if denom not in (0.0, None) and not np.isnan(denom) else 0.0

    if "syscv12" in X.columns:
        X.loc[:, "syscv12"] = _safe_cv("sys12std", "sys12mean")
    if "diacv12" in X.columns:
        X.loc[:, "diacv12"] = _safe_cv("dia12std", "dia12mean")

    if "circadiandysregulationindex" in X.columns:
        nondip = float(X["nondipperrisk"].iloc[0]) if "nondipperrisk" in X.columns else 0.0
        X.loc[:, "circadiandysregulationindex"] = (
            0.5 * float(X["syscv12"].iloc[0])
            + 0.3 * float(X["diacv12"].iloc[0])
            + 0.2 * nondip
        )

    X_t = supervised_bundle["preprocessor"].transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    iso_pred = int(iso.predict(X_t)[0])
    anomaly_score = float(iso.score_samples(X_t)[0])

    sy = float(latest["systolic"])
    di = float(latest["diastolic"])
    z_sys = (sy - patient_sys_mean) / (sys_scale + 1e-9)
    z_dia = (di - patient_dia_mean) / (dia_scale + 1e-9)

    # Primary personalized rule: same thresholds as the supervised target, applied to the
    # patient's own long-history baseline. This is what makes "unsupervised for long history"
    # genuinely personalized — a population IsolationForest alone can miss patient-specific spikes.
    row = pd.Series({
        "BPXOSY3": sy,
        "BPXODI3": di,
        "sys12mean": patient_sys_mean,
        "dia12mean": patient_dia_mean,
        "sys12std": patient_sys_std,
        "dia12std": patient_dia_std,
    })
    predicted_class = make_personalized_alert_type(row, sys_floor=sys_floor, dia_floor=dia_floor)
    if predicted_class is None:
        predicted_class = "Normal"

    return {
        "predicted_class": predicted_class,
        "probabilities": None,
        "details": {
            "isolation_forest_label": "anomaly" if iso_pred == -1 else "inlier",
            "anomaly_score": anomaly_score,
            "patient_sys_mean": patient_sys_mean,
            "patient_dia_mean": patient_dia_mean,
            "patient_sys_std": patient_sys_std,
            "patient_dia_std": patient_dia_std,
            "z_sys_latest": z_sys,
            "z_dia_latest": z_dia,
            "decision_source": "patient_baseline_z_rule",
        },
    }


def predict_patient_alert(
    payload: Dict[str, Any],
    supervised_bundle: Dict[str, Any],
    unsupervised_bundle: Dict[str, Any],
    threshold: int = C.UNSUPERVISED_READINGS_THRESHOLD,
) -> Dict[str, Any]:
    """Router: >threshold readings → unsupervised model, else supervised."""
    readings = payload.get("readings") or []
    n = len(readings)
    if n > threshold and unsupervised_bundle is not None and supervised_bundle is not None:
        result = predict_patient_alert_unsupervised(payload, supervised_bundle, unsupervised_bundle)
        result["model_used"] = "unsupervised"
    else:
        if supervised_bundle is None:
            raise ValueError("Supervised model not loaded.")
        result = predict_patient_alert_supervised(payload, supervised_bundle)
        result["model_used"] = "supervised"
        result.setdefault("details", {})
    result["n_readings"] = n
    return result
