---
title: HealPlace Cardio — BP Alert
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# HealPlace Cardio — BP Alert

Rule-based personalized blood-pressure alert engine implementing Dr. Manisha Singal's signed-off clinical specification (2025 AHA/ACC + CHAP trial + condition-specific heuristics).

**No ML.** Personalization comes from clinical context: age band, cardiac conditions, pregnancy, medications, and provider-configured targets. Alerts evaluate on the mean of the last 2–3 readings per session.

## Alert tiers

| Tier | Trigger | Push alert? |
|---|---|---|
| LEVEL_2 | sys ≥ 180 OR dia ≥ 120, OR any Level 2 symptom (severe headache, visual changes, altered mental status, chest pain, acute dyspnea, focal neuro deficit, severe epigastric/RUQ pain) | yes |
| LEVEL_1_HIGH | sys ≥ 160 OR dia ≥ 100 (condition/pregnancy-aware) | yes |
| STAGE_2 | 140–159 / 90–99 | dashboard only |
| STAGE_1 | 130–139 / 80–89 | dashboard only |
| ELEVATED | 120–129 and <80 | dashboard only |
| NORMAL | <120 and <80 | no |
| LEVEL_1_LOW | sys < age-adjusted floor OR dia < 60 | yes (patient safety) |

## Clinical context applied

- **Age bands** — 18–39 uses standard thresholds; 40–64 uses standard with comorbidity prompts; 65+ raises the low-BP floor from SBP 90 to SBP 100.
- **Cardiac conditions** — HFrEF/DCM (low SBP <85), HFpEF (low SBP <110), HCM (low SBP <100), CAD (diastolic alert <70). HFrEF and HCM require provider-configured thresholds.
- **Pregnancy** — LEVEL_1_HIGH at 140/90, LEVEL_2 at 160/110; ACE/ARB is a non-configurable contraindication alert.
- **Medications** — ACE/ARB + pregnancy (teratogenic), non-DHP CCB + HFrEF (negative inotropic), loop diuretic + SBP <90 (dashboard note), beta-blocker suppresses HR 50–60 alerts.
- **Pattern flags** — pulse pressure >60, morning surge ≥20 mmHg, non-dipping (night/day ratio >0.9), orthostatic drop ≥20/10.
- **AFib** — requires ≥3 readings per session before an alert.
- **Pre-measurement quality** — 8-point checklist; suboptimal readings still count but are flagged.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload          # UI at http://localhost:8000/
PYTHONPATH=. python tests/test_verification.py   # 22 verification scenarios
```

## API

- `GET /` — web UI
- `GET /api/health` — `{"status":"ok"}`
- `POST /api/predict` — PatientPayload → AlertResult (tier, messages, thresholds, pattern flags)
- `POST /api/clinical_context` — preview which thresholds would apply (no reading required)
- `POST /api/physician/target` / `GET /api/physician/target/{patient_id}` — per-patient target override (in-memory MVP)

Payload includes demographics, cardiac conditions, medications, Level 2 + pregnancy-specific symptoms, pre-measurement checklist, and a list of readings with optional `time_of_day` and `position`.

## Out of scope (will live in Cardioplace)

- Measurement gap alerts (requires DB + cron)
- Physician target persistence (currently in-memory for MVP)
- Aortic stenosis logic, post-pregnancy CV-risk flag (deferred)

## Deployment

Pushed to HuggingFace Spaces via [.github/workflows/deploy-huggingface.yml](.github/workflows/deploy-huggingface.yml). Docker image is small — no ML deps, no training step.
