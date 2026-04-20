"""
Rule-based alert constants derived from Dr. Manisha Singal's signed-off clinical
specification (2025 AHA/ACC + CHAP trial + condition-specific heuristics).

All thresholds here are HOME-CALIBRATED (office readings run ~5 mmHg higher).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# General adult thresholds (Section 2)
# ---------------------------------------------------------------------------

NORMAL_SYS_MAX: float = 120.0           # <120 and <80 = NORMAL
NORMAL_DIA_MAX: float = 80.0

ELEVATED_SYS_RANGE = (120.0, 129.0)     # <80 diastolic required for "Elevated"
STAGE_1_SYS_RANGE = (130.0, 139.0)
STAGE_1_DIA_RANGE = (80.0, 89.0)
STAGE_2_SYS_RANGE = (140.0, 159.0)
STAGE_2_DIA_RANGE = (90.0, 99.0)

# Level 1 High = Severe Stage 2 — PUSH alert threshold
LEVEL_1_HIGH_SYS: float = 160.0
LEVEL_1_HIGH_DIA: float = 100.0

# Level 2 = Hypertensive Emergency
LEVEL_2_SYS: float = 180.0
LEVEL_2_DIA: float = 120.0

# Level 1 Low — age-band specific (senior lower-bound bumped to 100)
LEVEL_1_LOW_SYS_YOUNG_MID: float = 90.0
LEVEL_1_LOW_SYS_SENIOR: float = 100.0
LEVEL_1_LOW_DIA: float = 60.0

# Age bands (Section 1)
AGE_BAND_MID_MIN: int = 40
AGE_BAND_SENIOR_MIN: int = 65

# ---------------------------------------------------------------------------
# Level 2 symptom overrides (Section 2.3)
# ---------------------------------------------------------------------------

LEVEL_2_SYMPTOMS = [
    "severe_headache_flag",
    "visual_changes_flag",
    "altered_mental_flag",
    "chest_pain_flag",
    "acute_dyspnea_flag",
    "focal_neuro_flag",
    "severe_epigastric_flag",
]

PREGNANCY_LEVEL_2_SYMPTOMS = [
    "new_headache_flag",
    "visual_changes_flag",
    "severe_epigastric_flag",  # RUQ pain
    "edema_flag",
]

# ---------------------------------------------------------------------------
# Pregnancy thresholds (Section 4)
# ---------------------------------------------------------------------------

PREGNANCY_LEVEL_1_HIGH_SYS: float = 140.0
PREGNANCY_LEVEL_1_HIGH_DIA: float = 90.0
PREGNANCY_LEVEL_2_SYS: float = 160.0
PREGNANCY_LEVEL_2_DIA: float = 110.0

# ---------------------------------------------------------------------------
# Condition-specific thresholds (Section 5)
# ---------------------------------------------------------------------------

HFREF_LOWER_SYS: float = 85.0
HFPEF_LOWER_SYS: float = 110.0
HCM_LOWER_SYS: float = 100.0
CAD_LOWER_DIA: float = 70.0

# Conditions that REQUIRE a provider-configured target before monitoring.
CONDITIONS_MANDATE_PROVIDER_CONFIG = ("has_hfref", "has_hcm", "has_dcm")

# ---------------------------------------------------------------------------
# Heart rate thresholds
# ---------------------------------------------------------------------------

AFIB_HR_HIGH: float = 110.0
AFIB_HR_LOW: float = 50.0
AFIB_MIN_READINGS: int = 3

TACHY_HR: float = 100.0
TACHY_CONSECUTIVE_READINGS: int = 2

BRADY_SYMPTOMATIC_HR: float = 50.0
BRADY_ASYMPTOMATIC_HR: float = 40.0
BETA_BLOCKER_SUPPRESSION_MIN: float = 50.0
BETA_BLOCKER_SUPPRESSION_MAX: float = 60.0

# ---------------------------------------------------------------------------
# Reading averaging (Section 6)
# ---------------------------------------------------------------------------

DEFAULT_AVERAGING_WINDOW: int = 3   # last N readings averaged for alert logic

# ---------------------------------------------------------------------------
# Pattern flags (already in use)
# ---------------------------------------------------------------------------

PULSE_PRESSURE_HIGH: float = 60.0
MORNING_SURGE_SYS_THRESHOLD: float = 20.0
NON_DIPPER_RATIO_THRESHOLD: float = 0.9
ORTHO_SYS_DROP: float = 20.0
ORTHO_DIA_DROP: float = 10.0
MED_TIMING_PEAK_HOURS: float = 2.0
MED_TIMING_TROUGH_HOURS: float = 20.0

# ---------------------------------------------------------------------------
# Provider target override heuristic (Section 5.1)
# ---------------------------------------------------------------------------

PROVIDER_TARGET_UPPER_HEURISTIC: float = 20.0   # Level 1 High = upper_target + 20

# ---------------------------------------------------------------------------
# Tier metadata (for response rendering)
# ---------------------------------------------------------------------------

TIER_ORDER = [
    "LEVEL_2", "LEVEL_1_HIGH", "STAGE_2", "STAGE_1",
    "ELEVATED", "NORMAL", "LEVEL_1_LOW",
]

TIER_COLORS = {
    "LEVEL_2": "red",
    "LEVEL_1_HIGH": "orange",
    "STAGE_2": "yellow",
    "STAGE_1": "blue",
    "ELEVATED": "lightblue",
    "NORMAL": "green",
    "LEVEL_1_LOW": "purple",
}

# Tiers that trigger a push notification to patient/caregiver/physician.
PUSH_ALERT_TIERS = {"LEVEL_2", "LEVEL_1_HIGH"}

# ---------------------------------------------------------------------------
# Age-band clinical context flags
# ---------------------------------------------------------------------------

AGE_BAND_CONTEXT = {
    "young":  "Lower baseline risk -- confirm sustained elevation before escalation.",
    "mid":    "Prompt comorbidity-specific threshold logic at onboarding.",
    "senior": "Assess for orthostatic symptoms and fall risk.",
}

# ---------------------------------------------------------------------------
# Contraindication alerts (Section 8)
# ---------------------------------------------------------------------------

CONTRAINDICATION_ACE_ARB_PREGNANCY = (
    "ACE inhibitors and ARBs are CONTRAINDICATED in pregnancy (teratogenic). "
    "Immediate provider notification required."
)
CONTRAINDICATION_NONDHP_CCB_HFREF = (
    "Non-dihydropyridine CCBs (diltiazem, verapamil) can be harmful in HFrEF "
    "due to negative inotropic effects."
)
