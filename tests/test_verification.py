"""22 verification scenarios for the rule-based alert engine."""
from __future__ import annotations

from src.alert_engine import evaluate


def _payload(**kw):
    """Build a PatientPayload dict with sensible defaults."""
    readings = kw.pop("readings", None) or [{"systolic": 120, "diastolic": 78, "pulse": 72}]
    base = {
        "age": 35, "gender": 2, "is_pregnant": 0,
        "has_hypertension": 0, "has_hfref": 0, "has_hfpef": 0, "has_hcm": 0,
        "has_dcm": 0, "has_cad": 0, "has_afib": 0, "has_tachycardia": 0, "has_bradycardia": 0,
        "on_ace_or_arb": 0, "on_beta_blocker": 0, "on_loop_diuretic": 0, "on_nondhp_ccb": 0,
        "hours_since_bp_med": None,
        "severe_headache_flag": 0, "visual_changes_flag": 0, "altered_mental_flag": 0,
        "chest_pain_flag": 0, "acute_dyspnea_flag": 0, "focal_neuro_flag": 0,
        "severe_epigastric_flag": 0, "new_headache_flag": 0, "edema_flag": 0,
        "dizziness_flag": 0, "patient_id": None,
        "measurement_conditions": {
            "no_caffeine": True, "no_smoking": True, "no_exercise": True,
            "bladder_empty": True, "seated_5min": True, "proper_posture": True,
            "not_talking": True, "cuff_bare_arm": True,
        },
        "readings": readings,
    }
    base.update(kw)
    return base


def _reading(s, d, p=72, **extra):
    return {"systolic": s, "diastolic": d, "pulse": p, **extra}


# --- Scenarios ---

def test_01_young_normal():
    r = evaluate(_payload(readings=[_reading(118, 76)]))
    assert r.tier == "NORMAL" and not r.push_alert


def test_02_young_stage2_dashboard_only():
    r = evaluate(_payload(readings=[_reading(145, 92)]))
    assert r.tier == "STAGE_2" and not r.push_alert


def test_03_young_level_1_high():
    r = evaluate(_payload(readings=[_reading(165, 102)]))
    assert r.tier == "LEVEL_1_HIGH" and r.push_alert


def test_04_young_level_2_by_number():
    r = evaluate(_payload(readings=[_reading(185, 125)]))
    assert r.tier == "LEVEL_2" and r.push_alert


def test_05_young_chest_pain_symptom_override():
    r = evaluate(_payload(readings=[_reading(130, 82)], chest_pain_flag=1))
    assert r.tier == "LEVEL_2" and r.push_alert


def test_06_young_severe_headache_symptom_override():
    r = evaluate(_payload(readings=[_reading(128, 80)], severe_headache_flag=1))
    assert r.tier == "LEVEL_2"


def test_07_senior_low_threshold_100():
    r = evaluate(_payload(age=70, readings=[_reading(95, 62)]))
    assert r.tier == "LEVEL_1_LOW"


def test_08_young_at_95_62_still_normal():
    r = evaluate(_payload(age=35, readings=[_reading(95, 62)]))
    assert r.tier == "NORMAL"


def test_09_pregnant_level_1_high_at_142_92():
    r = evaluate(_payload(age=30, is_pregnant=1, readings=[_reading(142, 92)]))
    assert r.tier == "LEVEL_1_HIGH"
    assert "OB" in r.messages["patient"] or "ob" in r.messages["patient"].lower()


def test_10_pregnant_level_2_at_165_112():
    r = evaluate(_payload(age=30, is_pregnant=1, readings=[_reading(165, 112)]))
    assert r.tier == "LEVEL_2"


def test_11_pregnant_ace_arb_contraindication_bumps_to_level2():
    r = evaluate(_payload(age=30, is_pregnant=1, on_ace_or_arb=1, readings=[_reading(118, 76)]))
    assert r.tier == "LEVEL_2"
    assert any("teratogenic" in c.lower() for c in r.contraindication_alerts)


def test_12_hfref_normal_at_88():
    r = evaluate(_payload(age=55, has_hfref=1, readings=[_reading(88, 60)]))
    assert r.tier == "NORMAL"


def test_13_hfref_low_below_85():
    r = evaluate(_payload(age=55, has_hfref=1, readings=[_reading(82, 58)]))
    assert r.tier == "LEVEL_1_LOW"


def test_14_hcm_low_below_100():
    r = evaluate(_payload(age=45, has_hcm=1, readings=[_reading(98, 70)]))
    assert r.tier == "LEVEL_1_LOW"


def test_15_cad_dia_caution_below_70():
    r = evaluate(_payload(age=60, has_cad=1, readings=[_reading(150, 68)]))
    assert r.tier == "LEVEL_1_LOW"
    assert any("CAD" in f for f in r.contributing_factors)


def test_16_afib_two_readings_insufficient():
    r = evaluate(_payload(age=65, has_afib=1, readings=[_reading(135, 85), _reading(137, 86)]))
    assert r.tier == "NORMAL"
    assert r.insufficient_readings is True


def test_17_afib_three_readings_hr_high():
    r = evaluate(_payload(age=65, has_afib=1, readings=[
        _reading(130, 80, 115), _reading(132, 82, 118), _reading(128, 78, 112),
    ]))
    assert r.hr_alert == "AFIB_HR_HIGH"


def test_18_bradycardia_hr55_on_betablocker_suppressed():
    r = evaluate(_payload(age=55, on_beta_blocker=1, readings=[
        _reading(120, 78, 55), _reading(122, 80, 55), _reading(120, 78, 55),
    ]))
    assert r.hr_alert != "BRADYCARDIA"


def test_19_bradycardia_hr55_no_betablocker_with_dizziness():
    r = evaluate(_payload(age=55, dizziness_flag=1, readings=[
        _reading(120, 78, 55), _reading(122, 80, 55), _reading(120, 78, 55),
    ]))
    assert r.hr_alert == "BRADYCARDIA"


def test_20_measurement_suboptimal_flag_only():
    payload = _payload(age=35, readings=[_reading(128, 82)])
    payload["measurement_conditions"]["no_caffeine"] = False
    r = evaluate(payload)
    assert r.measurement_quality == "suboptimal"
    # Tier still evaluates normally.
    assert r.tier in {"STAGE_1", "ELEVATED", "NORMAL"}


def test_21_three_readings_averaged():
    r = evaluate(_payload(readings=[_reading(120, 78), _reading(130, 82), _reading(140, 88)]))
    assert r.applied_reading["n_averaged"] == 3
    assert abs(r.applied_reading["systolic"] - 130.0) < 0.1


def test_22_provider_target_upper_plus_20():
    target = {"target_sys": 120, "target_dia": 75}
    r = evaluate(_payload(age=55, patient_id="p1", readings=[_reading(145, 88)]), physician_target=target)
    # Level 1 High = 120 + 20 = 140; 145 >= 140 → LEVEL_1_HIGH
    assert r.tier == "LEVEL_1_HIGH"


if __name__ == "__main__":
    import sys
    import traceback
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
    tests.sort()
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  OK  {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} scenarios pass")
    sys.exit(0 if failed == 0 else 1)
