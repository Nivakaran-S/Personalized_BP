"""
Thin MLflow tracking helper for the BP pipeline.

Reads MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD from the
environment (loaded from .env if present). If MLFLOW_TRACKING_URI is unset, all tracking
calls become no-ops so training still works offline / on HuggingFace Spaces without extra
config.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

from src.logging.logger import logging

# MLflow 3.x prints emoji status lines during logging; Windows' default cp1252
# stdout can't encode them. Reconfigure stdout/stderr to UTF-8 if possible.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    _MLFLOW_AVAILABLE = False


_CONFIGURED = False
_ENABLED = False


def _configure_once() -> bool:
    """Lazy one-time configuration; returns True if MLflow is ready to use."""
    global _CONFIGURED, _ENABLED
    if _CONFIGURED:
        return _ENABLED
    _CONFIGURED = True
    if not _MLFLOW_AVAILABLE:
        logging.info("mlflow not installed — skipping experiment tracking.")
        return False
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        logging.info("MLFLOW_TRACKING_URI not set — skipping experiment tracking.")
        return False
    mlflow.set_tracking_uri(uri)
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "bp-personalized-alert")
    try:
        mlflow.set_experiment(experiment)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.set_experiment failed (%s) — tracking disabled.", exc)
        return False
    _ENABLED = True
    logging.info("MLflow tracking enabled: uri=%s experiment=%s", uri, experiment)
    return True


def is_enabled() -> bool:
    return _configure_once()


@contextmanager
def start_run(run_name: str, nested: bool = False, tags: Optional[Dict[str, str]] = None) -> Iterator[Any]:
    """Start an MLflow run if tracking is enabled; otherwise yield None.

    If mlflow.start_run() fails at *entry* (auth, connection), degrade to a no-op
    so training still succeeds. Errors inside the body propagate normally — the
    outer mlflow context handles cleanup and re-raise.
    """
    if not is_enabled():
        yield None
        return
    try:
        run_cm = mlflow.start_run(run_name=run_name, nested=nested, tags=tags or {})
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.start_run(%s) failed to start: %s", run_name, exc)
        yield None
        return
    with run_cm as run:
        yield run


def log_params(params: Dict[str, Any]) -> None:
    if not is_enabled():
        return
    try:
        # MLflow limits param values to strings of length 500; stringify + truncate.
        safe = {k: str(v)[:500] for k, v in params.items() if v is not None}
        if safe:
            mlflow.log_params(safe)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.log_params failed: %s", exc)


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    if not is_enabled():
        return
    try:
        numeric = {k: float(v) for k, v in metrics.items() if v is not None and _is_numeric(v)}
        if numeric:
            mlflow.log_metrics(numeric, step=step)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.log_metrics failed: %s", exc)


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    if not is_enabled() or not path or not os.path.exists(path):
        return
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.log_artifact(%s) failed: %s", path, exc)


def log_artifacts(dir_path: str, artifact_path: Optional[str] = None) -> None:
    if not is_enabled() or not dir_path or not os.path.isdir(dir_path):
        return
    try:
        mlflow.log_artifacts(dir_path, artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.log_artifacts(%s) failed: %s", dir_path, exc)


def log_sklearn_model(model, artifact_path: str, registered_model_name: Optional[str] = None) -> None:
    if not is_enabled():
        return
    try:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, registered_model_name=registered_model_name)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.sklearn.log_model(%s) failed: %s", artifact_path, exc)


def log_dict(obj: Dict[str, Any], artifact_path: str) -> None:
    if not is_enabled():
        return
    try:
        mlflow.log_dict(obj, artifact_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.log_dict(%s) failed: %s", artifact_path, exc)


def set_tags(tags: Dict[str, str]) -> None:
    if not is_enabled():
        return
    try:
        mlflow.set_tags({k: str(v) for k, v in tags.items() if v is not None})
    except Exception as exc:  # noqa: BLE001
        logging.warning("mlflow.set_tags failed: %s", exc)


def _is_numeric(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
