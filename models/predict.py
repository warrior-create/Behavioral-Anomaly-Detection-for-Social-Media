"""
Inference / Prediction Module  — models/predict.py
====================================================
Provides a complete inference pipeline for scoring NEW users/posts
using all trained models (supervised + unsupervised).

Previously: completely missing — no way to run inference on new data.

This module:
  1. Loads all saved model artefacts (models, scaler, PCA, feature names)
  2. Accepts raw text + metadata DataFrame of new posts
  3. Runs the full feature extraction pipeline
  4. Produces a bot-probability score and final verdict per user
  5. Exports results as a ranked CSV
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default paths — can be overridden via constructor
DEFAULT_MODELS_DIR   = "data/models"
DEFAULT_FEATURES_DIR = "data/features"
DEFAULT_OUTPUT_DIR   = "data/results"


class BotPredictor:
    """
    End-to-end inference pipeline for CIB / bot detection.

    Usage:
        predictor = BotPredictor()
        predictor.load_artefacts()
        results = predictor.predict_from_csv("new_data.csv")
        print(results.head())
    """

    def __init__(
        self,
        models_dir:   str = DEFAULT_MODELS_DIR,
        features_dir: str = DEFAULT_FEATURES_DIR,
        output_dir:   str = DEFAULT_OUTPUT_DIR,
        config:       Optional[Dict] = None,
    ):
        self.models_dir   = models_dir
        self.features_dir = features_dir
        self.output_dir   = output_dir
        self.config       = config or {}

        # Loaded artefacts
        self.supervised_models: Dict   = {}
        self.scaler                    = None
        self.pca                       = None
        self.feature_names: List[str]  = []
        self._artefacts_loaded         = False

        logger.info("BotPredictor initialised | models_dir=%s", models_dir)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_artefacts(self):
        """Load all trained model artefacts from disk."""
        # Feature names
        fn_path = os.path.join(self.models_dir, "feature_names.pkl")
        if os.path.exists(fn_path):
            self.feature_names = joblib.load(fn_path)
            logger.info("Loaded %d feature names.", len(self.feature_names))

        # Scaler
        sc_path = os.path.join(self.models_dir, "scaler.pkl")
        if os.path.exists(sc_path):
            self.scaler = joblib.load(sc_path)
            logger.info("Loaded scaler.")

        # PCA
        pca_path = os.path.join(self.models_dir, "pca.pkl")
        if os.path.exists(pca_path):
            self.pca = joblib.load(pca_path)
            logger.info("Loaded PCA.")

        # Supervised models
        _SKIP = {
            "scaler.pkl", "pca.pkl", "feature_names.pkl",
            "lda_model.pkl", "lda_vectorizer.pkl",
        }
        _UNSUPERVISED = {"dbscan", "kmeans", "isolation_forest", "gmm", "hdbscan"}

        for pkl_file in sorted(os.listdir(self.models_dir)):
            if not pkl_file.endswith(".pkl") or pkl_file in _SKIP:
                continue
            model_name = pkl_file.replace(".pkl", "")
            if model_name in _UNSUPERVISED:
                continue
            try:
                model = joblib.load(os.path.join(self.models_dir, pkl_file))
                self.supervised_models[model_name] = model
                logger.info("Loaded model: %s", model_name)
            except Exception as exc:
                logger.error("Could not load %s: %s", pkl_file, exc)

        self._artefacts_loaded = True
        logger.info(
            "Artefacts loaded | %d supervised models.", len(self.supervised_models)
        )

    def predict_from_csv(
        self,
        raw_data_path: str,
        output_filename: str = "prediction_results.csv",
        primary_model: str = "random_forest",
    ) -> pd.DataFrame:
        """
        Full prediction pipeline from a raw CSV of posts.

        Pipeline:
          1. Clean text
          2. Extract all NLP + stylometric features
          3. Combine features
          4. Run all supervised models → ensemble probability
          5. Return ranked DataFrame

        Args:
            raw_data_path:   Path to CSV with at minimum 'user_id' and 'text'.
            output_filename: Name for the results CSV in output_dir.
            primary_model:   Model to use for final verdict (default rf).

        Returns:
            DataFrame with user_id, bot probability, verdict, and per-model scores.
        """
        if not self._artefacts_loaded:
            self.load_artefacts()

        logger.info("Predicting from: %s", raw_data_path)
        raw_df = pd.read_csv(raw_data_path)

        # ----- Step 1: Extract features -----
        feature_df = self._run_feature_extraction(raw_df)

        if feature_df.empty:
            logger.error("Feature extraction returned empty DataFrame.")
            return pd.DataFrame()

        # ----- Step 2: Align to training feature names -----
        X = self._align_features(feature_df)

        # ----- Step 3: Score with each supervised model -----
        results = pd.DataFrame({"user_id": feature_df["user_id"]})

        proba_cols = []
        for model_name, model in self.supervised_models.items():
            try:
                X_model = self._align_to_model(feature_df, model)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_model)[:, 1]
                else:
                    pred = model.predict(X_model)
                    proba = pred.astype(float)

                col = f"proba_{model_name}"
                results[col] = proba
                proba_cols.append(col)
                results[f"pred_{model_name}"] = (proba >= 0.5).astype(int)
            except Exception as exc:
                logger.warning("Prediction failed for %s: %s", model_name, exc)

        # ----- Step 4: Ensemble score (mean of all probabilities) -----
        if proba_cols:
            results["bot_probability"] = results[proba_cols].mean(axis=1)
            results["bot_verdict"]     = (results["bot_probability"] >= 0.5).astype(int)
            results["confidence"]      = results["bot_probability"].apply(
                lambda p: "HIGH" if abs(p - 0.5) > 0.3 else
                          "MEDIUM" if abs(p - 0.5) > 0.15 else "LOW"
            )
        else:
            logger.error("No model produced predictions.")
            return pd.DataFrame()

        # ----- Step 5: Sort by bot_probability descending -----
        results = results.sort_values("bot_probability", ascending=False)

        # ----- Step 6: Save -----
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, output_filename)
        results.to_csv(out_path, index=False)
        logger.info("Results saved → %s", out_path)

        self._print_prediction_summary(results)
        return results

    def predict_single_user(
        self,
        user_id: str,
        texts: List[str],
        timestamps: Optional[List[str]] = None,
    ) -> Dict:
        """
        Score a single user in-memory (useful for real-time scoring).

        Args:
            user_id:    User identifier.
            texts:      List of post texts.
            timestamps: Optional list of ISO timestamp strings.

        Returns:
            Dict with bot_probability, verdict, confidence, per-model scores.
        """
        if not self._artefacts_loaded:
            self.load_artefacts()

        # Build minimal DataFrame
        n = len(texts)
        data = {
            "user_id":   [user_id] * n,
            "text":      texts,
            "post_id":   [f"{user_id}_{i}" for i in range(n)],
        }
        if timestamps:
            data["timestamp"] = timestamps[:n]

        df = pd.DataFrame(data)
        feature_df = self._run_feature_extraction(df)

        if feature_df.empty:
            return {"error": "Feature extraction failed."}

        scores = {}
        for model_name, model in self.supervised_models.items():
            try:
                X_model = self._align_to_model(feature_df, model)
                if hasattr(model, "predict_proba"):
                    scores[model_name] = float(model.predict_proba(X_model)[0, 1])
                else:
                    scores[model_name] = float(model.predict(X_model)[0])
            except Exception as exc:
                logger.debug("Single-user prediction failed for %s: %s", model_name, exc)

        if not scores:
            return {"error": "All models failed to predict."}

        ensemble = float(np.mean(list(scores.values())))
        return {
            "user_id":        user_id,
            "bot_probability": ensemble,
            "bot_verdict":    1 if ensemble >= 0.5 else 0,
            "confidence":     "HIGH" if abs(ensemble - 0.5) > 0.3 else
                              "MEDIUM" if abs(ensemble - 0.5) > 0.15 else "LOW",
            "per_model_scores": scores,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _run_feature_extraction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the same feature extraction steps as the training pipeline.
        Uses only features that don't require a separate corpus fit
        (stylometry, temporal, perplexity with self-reference LM).
        """
        try:
            import tempfile, os as _os

            # Save temp CSV
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name
                df.to_csv(tmp_path, index=False)

            parts = []

            # 1. Stylometric features
            try:
                from preprocessing.stylometry import StylometricAnalyzer
                stylo = StylometricAnalyzer()
                stylo_feat = stylo.extract_features(tmp_path)
                if not stylo_feat.empty:
                    parts.append(stylo_feat)
            except Exception as exc:
                logger.warning("Stylometry failed: %s", exc)

            # 2. TF-IDF features
            try:
                from features.tfidf_features import TFIDFAnalyzer
                tfidf = TFIDFAnalyzer({"min_df": 1})
                tfidf_feat = tfidf.extract_features(tmp_path)
                if not tfidf_feat.empty:
                    parts.append(tfidf_feat)
            except Exception as exc:
                logger.warning("TF-IDF failed: %s", exc)

            # 3. PMI features
            try:
                from features.pmi_cooccurrence import PMICooccurrenceAnalyzer
                pmi = PMICooccurrenceAnalyzer({"min_freq": 1})
                pmi_feat = pmi.extract_features(tmp_path)
                if not pmi_feat.empty:
                    parts.append(pmi_feat)
            except Exception as exc:
                logger.warning("PMI failed: %s", exc)

            # 4. Perplexity features
            try:
                from features.ngram_perplexity import PerplexityFeatureExtractor
                px = PerplexityFeatureExtractor()
                px_feat = px.extract_features(tmp_path)
                if not px_feat.empty:
                    parts.append(px_feat)
            except Exception as exc:
                logger.warning("Perplexity failed: %s", exc)

            # 5. Temporal features (only if timestamp column present)
            if "timestamp" in df.columns:
                try:
                    from features.temporal_features import TemporalFeatureExtractor
                    temporal = TemporalFeatureExtractor()
                    temp_feat = temporal.extract_features(tmp_path)
                    if not temp_feat.empty:
                        parts.append(temp_feat)
                except Exception as exc:
                    logger.warning("Temporal features failed: %s", exc)

            _os.unlink(tmp_path)

            if not parts:
                return pd.DataFrame()

            # Merge all on user_id
            merged = parts[0]
            for part in parts[1:]:
                overlap = [c for c in part.columns if c in merged.columns and c != "user_id"]
                if overlap:
                    part = part.drop(columns=overlap)
                merged = pd.merge(merged, part, on="user_id", how="outer")

            merged = merged.fillna(0)
            merged["user_id"] = merged["user_id"].astype(str).str.strip()
            return merged

        except Exception as exc:
            logger.error("Feature extraction pipeline failed: %s", exc)
            return pd.DataFrame()

    def _align_to_model(self, feature_df: pd.DataFrame, model):
        """Return feature DataFrame aligned to what this specific model expects."""
        if hasattr(model, "feature_names_in_"):
            required = list(model.feature_names_in_)
        elif self.feature_names:
            required = self.feature_names
        else:
            drop = ["user_id"]
            return feature_df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number]).fillna(0)

        for col in required:
            if col not in feature_df.columns:
                feature_df[col] = 0

        return feature_df[required].fillna(0)

    def _align_features(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Align to saved global feature names."""
        if self.feature_names:
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            return feature_df[self.feature_names].fillna(0).values
        return feature_df.select_dtypes(include=[np.number]).fillna(0).values

    @staticmethod
    def _print_prediction_summary(results: pd.DataFrame):
        """Print a human-readable prediction summary."""
        n_total = len(results)
        n_bots  = int((results["bot_verdict"] == 1).sum())
        n_human = n_total - n_bots
        high_conf = int((results["confidence"] == "HIGH").sum())

        print("\n" + "=" * 60)
        print("  BOT DETECTION RESULTS")
        print("=" * 60)
        print(f"  Total users analysed: {n_total}")
        print(f"  Detected bots:        {n_bots}  ({n_bots/max(n_total,1):.1%})")
        print(f"  Likely human:         {n_human} ({n_human/max(n_total,1):.1%})")
        print(f"  High-confidence:      {high_conf}")
        print("=" * 60)

        print("\n  Top 10 suspected bots:")
        cols = ["user_id", "bot_probability", "bot_verdict", "confidence"]
        top10 = results.head(10)[cols]
        print(top10.to_string(index=False))
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_new_data.csv>")
        print("       python predict.py data/raw/new_tweets.csv")
        sys.exit(1)

    predictor = BotPredictor()
    predictor.load_artefacts()
    results = predictor.predict_from_csv(sys.argv[1])
    print(f"\nPredictions complete. {len(results)} users scored.")
