"""Run the full BP training pipeline end-to-end against local NHANES/Dryad data."""
from __future__ import annotations

import sys

from src.exception.exception import BPException
from src.logging.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline


if __name__ == "__main__":
    try:
        logging.info("Launching BP training pipeline.")
        artifact = TrainingPipeline().run_pipeline()
        logging.info("Pipeline finished. Best model: %s", artifact.best_model_name)
        print("Best model:", artifact.best_model_name)
        print("Test metrics:", artifact.test_metrics)
        print("Bundle:", artifact.model_bundle_file_path)
    except Exception as exc:  # noqa: BLE001
        raise BPException(exc, sys)
