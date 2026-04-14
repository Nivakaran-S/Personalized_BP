"""
Load NHANES XPT files and (optional) Dryad supplementary data from the local data/ folder,
merge into a single feature-store CSV, and compute Dryad population statistics used
downstream by the transformation stage. Replaces the previous MongoDB-backed ingestion.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.constants import training_pipeline as C
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception.exception import BPException
from src.logging.logger import logging


NHANES_CDC_URL_TEMPLATE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{file}"
NHANES_MIN_VALID_BYTES = 100_000


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    @staticmethod
    def _decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].apply(lambda v: v.decode("latin1") if isinstance(v, bytes) else v)
        return df

    @staticmethod
    def _looks_like_valid_xpt(path: str) -> bool:
        """Quick sanity check: real NHANES XPT files are >100 KB and start with the XPORT header."""
        if not os.path.exists(path):
            return False
        try:
            if os.path.getsize(path) < NHANES_MIN_VALID_BYTES:
                return False
            with open(path, "rb") as f:
                header = f.read(40)
            return b"HEADER RECORD" in header
        except OSError:
            return False

    def _download_nhanes_file(self, file_name: str, dest: str) -> None:
        url = NHANES_CDC_URL_TEMPLATE.format(file=file_name)
        logging.info("Downloading NHANES file: %s", url)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (BP-pipeline auto-fetch)"}
        )
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        size = os.path.getsize(dest)
        logging.info("  -> %s: %d bytes", file_name, size)
        if not self._looks_like_valid_xpt(dest):
            raise BPException(
                ValueError(
                    f"Downloaded {file_name} ({size} bytes) is not a valid XPT file — "
                    "the CDC endpoint may be blocked from this host."
                ),
                sys,
            )

    def _ensure_nhanes_file(self, file_name: str) -> str:
        path = os.path.join(self.data_ingestion_config.nhanes_dir, file_name)
        if self._looks_like_valid_xpt(path):
            return path
        logging.info("NHANES file missing or invalid (%s) — attempting download.", path)
        self._download_nhanes_file(file_name, path)
        return path

    def _load_xpt(self, file_name: str) -> pd.DataFrame:
        path = self._ensure_nhanes_file(file_name)
        try:
            df = pd.read_sas(path, format="xport", encoding="latin1")
        except ValueError as exc:
            raise BPException(
                ValueError(
                    f"{path} is not a valid SAS XPORT file ({exc}). "
                    "Replace it with a real NHANES XPT download."
                ),
                sys,
            )
        return self._decode_bytes(df)

    def load_nhanes(self) -> pd.DataFrame:
        try:
            bp = self._load_xpt(self.data_ingestion_config.nhanes_bpxo_file)
            demo = self._load_xpt(self.data_ingestion_config.nhanes_demo_file)
            bmx = self._load_xpt(self.data_ingestion_config.nhanes_bmx_file)
            rx = self._load_xpt(self.data_ingestion_config.nhanes_rxq_file)
            logging.info(
                "Loaded NHANES: bp=%s demo=%s bmx=%s rx=%s",
                bp.shape, demo.shape, bmx.shape, rx.shape,
            )

            rx_agg = self._aggregate_rx(rx)

            merged = demo.merge(bp, on="SEQN", how="inner")
            merged = merged.merge(bmx, on="SEQN", how="left")
            merged = merged.merge(rx_agg, on="SEQN", how="left")
            merged["antihypertensiveflag"] = merged["antihypertensiveflag"].fillna(0).astype(int)
            merged["rxcount"] = merged["rxcount"].fillna(0)
            logging.info("Merged NHANES shape: %s", merged.shape)
            return merged
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)

    @staticmethod
    def _aggregate_rx(rx: pd.DataFrame) -> pd.DataFrame:
        """Per-subject antihypertensive flag + Rx count (from RXDCOUNT/RXQ050/RXQ033 if present)."""
        if rx.empty or "SEQN" not in rx.columns:
            return pd.DataFrame(columns=["SEQN", "antihypertensiveflag", "rxcount"])
        flag_col = "RXDUSE" if "RXDUSE" in rx.columns else None
        count_col = next((c for c in ["RXDCOUNT", "RXQ050", "RXQ033"] if c in rx.columns), None)
        grouped = rx.groupby("SEQN", as_index=False).agg(
            antihypertensiveflag=(flag_col, "max") if flag_col else ("SEQN", "size"),
            rxcount=(count_col, "max") if count_col else ("SEQN", "size"),
        )
        if not flag_col:
            grouped["antihypertensiveflag"] = 0
        grouped["antihypertensiveflag"] = pd.to_numeric(
            grouped["antihypertensiveflag"], errors="coerce"
        ).fillna(0).astype(int).clip(0, 1)
        grouped["rxcount"] = pd.to_numeric(grouped["rxcount"], errors="coerce").fillna(0)
        return grouped

    def compute_dryad_stats(self) -> Dict[str, float]:
        """Compute population dipping / morning-surge / std statistics from Dryad, if files exist."""
        sleep_path = os.path.join(self.data_ingestion_config.dryad_dir, self.data_ingestion_config.dryad_sleep_file)
        if not os.path.exists(sleep_path):
            logging.info("Dryad sleep file missing (%s); using DRYAD_DEFAULTS.", sleep_path)
            return dict(C.DRYAD_DEFAULTS)
        try:
            df = pd.read_excel(sleep_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not read Dryad sleep file: %s; using defaults.", exc)
            return dict(C.DRYAD_DEFAULTS)

        sys_cols = [c for c in df.columns if "sys" in c.lower()]
        dia_cols = [c for c in df.columns if "dia" in c.lower()]
        if not sys_cols or not dia_cols:
            return dict(C.DRYAD_DEFAULTS)

        sys_vals = df[sys_cols].apply(pd.to_numeric, errors="coerce")
        dia_vals = df[dia_cols].apply(pd.to_numeric, errors="coerce")
        sys_vals = sys_vals.where((sys_vals >= C.SYS_VALID_RANGE[0]) & (sys_vals <= C.SYS_VALID_RANGE[1]))
        dia_vals = dia_vals.where((dia_vals >= C.DIA_VALID_RANGE[0]) & (dia_vals <= C.DIA_VALID_RANGE[1]))

        sys_std = sys_vals.std(axis=1, ddof=0)
        dia_std = dia_vals.std(axis=1, ddof=0)

        valid_sys_std = sys_std[(sys_std >= 1) & (sys_std <= 30)]
        valid_dia_std = dia_std[(dia_std >= 1) & (dia_std <= 30)]

        half = len(sys_cols) // 2 or 1
        morning_mean = sys_vals.iloc[:, :half].mean(axis=1)
        evening_mean = sys_vals.iloc[:, half:].mean(axis=1)
        morning_evening_diff = (morning_mean - evening_mean)
        morning_evening_diff = morning_evening_diff[
            (morning_evening_diff >= -40) & (morning_evening_diff <= 40)
        ]

        dipping_ratio = evening_mean / morning_mean
        dipping_ratio = dipping_ratio[(dipping_ratio >= 0.5) & (dipping_ratio <= 1.2)]

        stats = {
            "meanmorningeveningdiff": float(morning_evening_diff.mean())
            if not morning_evening_diff.empty else C.DRYAD_DEFAULTS["meanmorningeveningdiff"],
            "meandippingratio": float(dipping_ratio.mean())
            if not dipping_ratio.empty else C.DRYAD_DEFAULTS["meandippingratio"],
            "meansysstd": float(valid_sys_std.mean())
            if not valid_sys_std.empty else C.DRYAD_DEFAULTS["meansysstd"],
            "meandiastd": float(valid_dia_std.mean())
            if not valid_dia_std.empty else C.DRYAD_DEFAULTS["meandiastd"],
        }
        # Replace NaNs with defaults defensively.
        for k, v in stats.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                stats[k] = C.DRYAD_DEFAULTS[k]
        return stats

    def _write_outputs(self, df: pd.DataFrame, dryad_stats: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)
        df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
        with open(self.data_ingestion_config.dryad_stats_file_path, "w") as f:
            json.dump(dryad_stats, f, indent=2)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion from local NHANES + Dryad files.")
            merged = self.load_nhanes()
            dryad_stats = self.compute_dryad_stats()
            self._write_outputs(merged, dryad_stats)
            artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                dryad_stats_file_path=self.data_ingestion_config.dryad_stats_file_path,
            )
            logging.info("Data ingestion artifact: %s", artifact)
            return artifact
        except BPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BPException(exc, sys)
