"""
Feelow Backend — Multi-Model Sentiment Engine
No Streamlit dependency. Uses instance-level dict caching for model pipelines.
"""

from __future__ import annotations
import os, logging
from typing import List, Dict, Optional

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"

import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    Pipeline,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS, DEFAULT_MODEL_ID, SENTIMENT_MAP

logger = logging.getLogger(__name__)


class MultiModelSentimentEngine:
    """
    Financial sentiment engine with multi-model ensemble capability.
    Model pipelines are cached in self._pipelines dict (no Streamlit needed).
    """

    def __init__(self):
        self.device = self._detect_device()
        self._pipelines: Dict[str, Pipeline] = {}

    @staticmethod
    def _detect_device() -> int:
        if torch.cuda.is_available():
            return 0
        return -1

    @property
    def device_name(self) -> str:
        return "CUDA GPU" if self.device == 0 else "CPU"

    def _load_pipeline(self, model_id: str) -> Pipeline:
        """Load a HuggingFace pipeline (cached in self._pipelines)."""
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        device_str = f"cuda:{self.device}" if self.device >= 0 else "cpu"
        return pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device_str,
            truncation=True,
            max_length=512,
        )

    def get_pipeline(self, model_id: str = DEFAULT_MODEL_ID) -> Pipeline:
        if model_id not in self._pipelines:
            logger.info(f"Loading sentiment model: {model_id}")
            self._pipelines[model_id] = self._load_pipeline(model_id)
        return self._pipelines[model_id]

    def analyze_headlines(
        self,
        headlines: List[str],
        model_id: str = DEFAULT_MODEL_ID,
    ) -> pd.DataFrame:
        if not headlines:
            return pd.DataFrame(columns=["label", "score", "sentiment_numeric"])
        try:
            clf = self.get_pipeline(model_id)
            results = clf(headlines)
            df = pd.DataFrame(results)
            df["label_raw"] = df["label"]
            df["label"] = df["label"].str.lower()
            df["sentiment_numeric"] = df["label_raw"].map(SENTIMENT_MAP).fillna(0).astype(int)
            return df
        except Exception as e:
            logger.error(f"Sentiment error ({model_id}): {e}")
            return pd.DataFrame(columns=["label", "score", "sentiment_numeric"])

    def analyze_ensemble(
        self,
        headlines: List[str],
        model_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if model_ids is None:
            model_ids = list(MODELS.keys())

        all_results = {}
        for mid in model_ids:
            try:
                res = self.analyze_headlines(headlines, mid)
                short = MODELS.get(mid, {}).get("name", mid).replace(" ", "_").replace("/", "_")
                all_results[f"{short}_label"] = res["label"].values
                all_results[f"{short}_score"] = res["score"].values
                all_results[f"{short}_numeric"] = res["sentiment_numeric"].values
            except Exception as e:
                logger.warning(f"Skipping {mid}: {e}")

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        df["headline"] = headlines

        numeric_cols = [c for c in df.columns if c.endswith("_numeric")]
        if numeric_cols:
            df["ensemble_score"] = df[numeric_cols].mean(axis=1)
            df["ensemble_label"] = df["ensemble_score"].apply(
                lambda x: "positive" if x > 0.33 else ("negative" if x < -0.33 else "neutral")
            )
        return df

    def compare_models(self, text: str) -> pd.DataFrame:
        rows = []
        for mid, meta in MODELS.items():
            try:
                clf = self.get_pipeline(mid)
                res = clf([text])[0]
                rows.append({
                    "Model": meta["name"],
                    "Label": res["label"],
                    "Confidence": res["score"],
                    "Numeric": SENTIMENT_MAP.get(res["label"], 0),
                })
            except Exception as e:
                rows.append({
                    "Model": meta["name"],
                    "Label": f"Error: {e}",
                    "Confidence": 0.0,
                    "Numeric": 0,
                })
        return pd.DataFrame(rows)
