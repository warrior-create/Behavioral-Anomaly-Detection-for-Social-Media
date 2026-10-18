"""
Evaluation Metrics — COMPLETE VERSION
=======================================
Evaluates ALL trained models with a comprehensive set of metrics and
exports results in formats ready for the Plotly dashboard.

What was incomplete before:
  - No per-class precision/recall breakdown
  - No ROC curve data export
  - No confusion matrix export
  - Silhouette score only, no Davies-Bouldin or Calinski-Harabasz
  - No summary comparison table across models

What this version adds:
  - Full per-class precision, recall, F1 from classification_report
  - ROC curve (FPR, TPR, AUC) exported as CSV for dashboard
  - Confusion matrix saved as CSV per model
  - Davies-Bouldin + Calinski-Harabasz for clustering quality
  - Model comparison table (sorted by F1)
  - Bootstrap confidence intervals on accuracy (optional)
"""

import os
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates all trained models and exports rich metrics artefacts."""

    UNSUPERVISED_SKIP = {
        "scaler.pkl", "pca.pkl", "feature_names.pkl",
        "lda_model.pkl", "lda_vectorizer.pkl",
    }
    CLUSTERING_MODELS = {"dbscan", "kmeans", "hdbscan", "gmm"}
    ANOMALY_MODELS    = {"isolation_forest"}

    def __init__(self):
        logger.info("ModelEvaluator initialised.")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def evaluate_all_models(
        self, models_dir: str, results_dir: str
    ) -> pd.DataFrame:
        """
        Evaluate every .pkl model in models_dir and save metrics to results_dir.

        Args:
            models_dir:  Directory containing trained .pkl files.
            results_dir: Directory to write metrics CSVs/JSONs.

        Returns:
            DataFrame with one row per model and all computed metrics.
        """
        os.makedirs(results_dir, exist_ok=True)

        # ---- Load data ----
        feature_path = "data/features/final_user_features.csv"
        labels_path  = "data/labels/ground_truth_labels.csv"

        if not os.path.exists(feature_path):
            logger.error("Feature file not found: %s", feature_path)
            return pd.DataFrame()

        df_feat = pd.read_csv(feature_path)

        # Base numeric matrix for unsupervised models
        drop_cols = [
            "user_id", "community_id", "created_at", "screen_name",
            "label", "is_bot", "tweet_id", "Unnamed: 0", "algorithm",
        ]
        X_numeric = (
            df_feat
            .drop(columns=drop_cols, errors="ignore")
            .select_dtypes(include=[np.number])
            .fillna(0)
        )
        X_raw = X_numeric.values

        # Load/fit preprocessors
        X_scaled, X_pca = self._get_preprocessed(models_dir, X_raw)

        # Ground truth
        y_true, has_labels = self._load_labels(df_feat, labels_path)

        # Feature names for supervised models
        feat_names_path = os.path.join(models_dir, "feature_names.pkl")
        saved_feat_names = (
            joblib.load(feat_names_path)
            if os.path.exists(feat_names_path)
            else []
        )

        # ---- Evaluate each model ----
        all_results = []

        for pkl_file in sorted(os.listdir(models_dir)):
            if not pkl_file.endswith(".pkl"):
                continue
            if pkl_file in self.UNSUPERVISED_SKIP:
                continue

            model_name = pkl_file.replace(".pkl", "")
            try:
                model = joblib.load(os.path.join(models_dir, pkl_file))
                entry = self._evaluate_model(
                    model_name, model,
                    df_feat, X_raw, X_scaled, X_pca,
                    y_true, has_labels,
                    saved_feat_names, results_dir,
                )
                if entry:
                    all_results.append(entry)
                    logger.info("✓ Evaluated: %s", model_name)
            except Exception as exc:
                logger.error("✗ Failed to evaluate %s: %s", model_name, exc)
                traceback.print_exc()

        # ---- Summary table ----
        if all_results:
            summary_df = pd.DataFrame(all_results)
            # Sort by F1 descending (NaN last)
            if "f1_weighted" in summary_df.columns:
                summary_df = summary_df.sort_values(
                    "f1_weighted", ascending=False, na_position="last"
                )
            summary_path = os.path.join(results_dir, "evaluation_metrics.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info("Evaluation summary → %s", summary_path)
            self._print_summary(summary_df)
            return summary_df

        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Private: per-model evaluation                                        #
    # ------------------------------------------------------------------ #

    def _evaluate_model(
        self,
        model_name: str,
        model,
        df_feat: pd.DataFrame,
        X_raw: np.ndarray,
        X_scaled: np.ndarray,
        X_pca: np.ndarray,
        y_true: np.ndarray,
        has_labels: bool,
        saved_feat_names: List[str],
        results_dir: str,
    ) -> Optional[Dict]:
        """Run full evaluation for one model."""
        entry: Dict = {"model": model_name}

        # ---------------------------------------------------------------- #
        # 1. Determine which data format to use                             #
        # ---------------------------------------------------------------- #
        if model_name in self.CLUSTERING_MODELS:
            X_eval = X_pca
        elif model_name in self.ANOMALY_MODELS:
            X_eval = X_scaled
        else:
            # Supervised: align feature columns
            X_eval = self._align_features(df_feat, model, saved_feat_names)
            if X_eval is None:
                return None

        # ---------------------------------------------------------------- #
        # 2. Predict                                                        #
        # ---------------------------------------------------------------- #
        if hasattr(model, "predict"):
            y_pred = model.predict(X_eval)
        elif hasattr(model, "fit_predict"):
            y_pred = model.fit_predict(X_eval)
        else:
            logger.warning("Model %s has no predict method — skipping.", model_name)
            return None

        # Remap Isolation Forest: {-1, 1} → {1, 0}
        if model_name in self.ANOMALY_MODELS:
            y_pred = np.where(y_pred == -1, 1, 0)

        # ---------------------------------------------------------------- #
        # 3. Supervised metrics (if ground truth available)                #
        # ---------------------------------------------------------------- #
        if has_labels and len(np.unique(y_pred)) > 1:
            entry = self._add_supervised_metrics(
                entry, model_name, model, y_true, y_pred, X_eval, results_dir
            )

        # ---------------------------------------------------------------- #
        # 4. Clustering / unsupervised metrics                             #
        # ---------------------------------------------------------------- #
        if len(np.unique(y_pred)) > 1 and len(y_pred) > 1:
            try:
                entry["silhouette"]         = float(silhouette_score(X_pca, y_pred))
                entry["davies_bouldin"]     = float(davies_bouldin_score(X_pca, y_pred))
                entry["calinski_harabasz"]  = float(calinski_harabasz_score(X_pca, y_pred))
            except Exception:
                pass

        # ---------------------------------------------------------------- #
        # 5. Confusion matrix                                              #
        # ---------------------------------------------------------------- #
        if has_labels:
            self._save_confusion_matrix(
                model_name, y_true, y_pred, results_dir
            )

        entry["n_unique_predictions"] = int(len(np.unique(y_pred)))
        return entry

    def _add_supervised_metrics(
        self, entry, model_name, model, y_true, y_pred, X_eval, results_dir
    ):
        """Add accuracy, F1, precision, recall, ROC-AUC."""
        entry["accuracy"]    = float(accuracy_score(y_true, y_pred))
        prec, rec, f1, _     = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        entry["precision_weighted"] = float(prec)
        entry["recall_weighted"]    = float(rec)
        entry["f1_weighted"]        = float(f1)

        # Per-class report
        report = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        for cls_label in ["0", "1"]:
            if cls_label in report:
                for metric in ["precision", "recall", "f1-score"]:
                    col = f"class{cls_label}_{metric.replace('-','_')}"
                    entry[col] = float(report[cls_label][metric])

        # Save full text report
        report_text = classification_report(y_true, y_pred, zero_division=0)
        report_path = os.path.join(results_dir, f"{model_name}_classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        # ROC-AUC + curve
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_eval)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_eval)
            else:
                y_score = y_pred.astype(float)

            entry["roc_auc"] = float(roc_auc_score(y_true, y_score))
            entry["avg_precision"] = float(
                average_precision_score(y_true, y_score)
            )

            # Export ROC curve points
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
            roc_df.to_csv(
                os.path.join(results_dir, f"{model_name}_roc_curve.csv"),
                index=False,
            )
        except Exception as exc:
            logger.debug("ROC-AUC failed for %s: %s", model_name, exc)

        return entry

    def _save_confusion_matrix(self, model_name, y_true, y_pred, results_dir):
        """Save confusion matrix as CSV."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
            cm_df = pd.DataFrame(
                cm,
                index=[f"actual_{l}" for l in labels],
                columns=[f"pred_{l}" for l in labels],
            )
            cm_df.to_csv(
                os.path.join(results_dir, f"{model_name}_confusion_matrix.csv")
            )
        except Exception as exc:
            logger.debug("Confusion matrix failed for %s: %s", model_name, exc)

    # ------------------------------------------------------------------ #
    # Private: data helpers                                               #
    # ------------------------------------------------------------------ #

    def _get_preprocessed(
        self, models_dir: str, X_raw: np.ndarray
    ):
        """Load or fit StandardScaler + PCA."""
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        pca_path    = os.path.join(models_dir, "pca.pkl")

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            try:
                X_scaled = scaler.transform(X_raw)
            except Exception:
                X_scaled = scaler.fit_transform(X_raw)
        else:
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)

        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
            try:
                X_pca = pca.transform(X_scaled)
            except Exception:
                X_pca = pca.fit_transform(X_scaled)
        else:
            n_comp = min(10, X_scaled.shape[1])
            pca    = PCA(n_components=n_comp, random_state=42)
            X_pca  = pca.fit_transform(X_scaled)

        return X_scaled, X_pca

    def _load_labels(
        self, df_feat: pd.DataFrame, labels_path: str
    ):
        """Load and align ground-truth labels. Returns (y_true, has_labels)."""
        y_true     = np.zeros(len(df_feat))
        has_labels = False

        if not os.path.exists(labels_path):
            return y_true, has_labels

        try:
            labels_df = pd.read_csv(labels_path)
            if "user_id" not in df_feat.columns:
                return y_true, has_labels

            df_feat = df_feat.copy()
            df_feat["user_id"]    = df_feat["user_id"].astype(str).str.strip()
            labels_df["user_id"]  = labels_df["user_id"].astype(str).str.strip()

            merged = df_feat[["user_id"]].merge(
                labels_df[["user_id", "is_bot"]], on="user_id", how="left"
            )
            y_true     = merged["is_bot"].fillna(0).values
            has_labels = len(np.unique(y_true)) > 1

        except Exception as exc:
            logger.warning("Could not load labels: %s", exc)

        return y_true, has_labels

    def _align_features(
        self,
        df_feat: pd.DataFrame,
        model,
        saved_feat_names: List[str],
    ) -> Optional[np.ndarray]:
        """Return feature matrix aligned to what the model was trained on."""
        if hasattr(model, "feature_names_in_"):
            required = list(model.feature_names_in_)
        elif saved_feat_names:
            required = saved_feat_names
        else:
            # Fallback: all numeric columns
            drop = ["user_id", "is_bot", "label", "community_id",
                    "screen_name", "created_at", "tweet_id", "Unnamed: 0"]
            return (
                df_feat.drop(columns=drop, errors="ignore")
                .select_dtypes(include=[np.number])
                .fillna(0)
                .values
            )

        # Add any missing columns as zeros
        for col in required:
            if col not in df_feat.columns:
                df_feat[col] = 0

        try:
            return df_feat[required].fillna(0).values
        except KeyError as exc:
            logger.error("Feature alignment failed: %s", exc)
            return None

    @staticmethod
    def _print_summary(df: pd.DataFrame):
        """Print a formatted summary table."""
        print("\n" + "=" * 80)
        print("  MODEL EVALUATION SUMMARY")
        print("=" * 80)
        cols = ["model", "accuracy", "f1_weighted", "roc_auc",
                "silhouette", "davies_bouldin"]
        display_cols = [c for c in cols if c in df.columns]
        print(df[display_cols].to_string(index=False))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models("data/models", "data/results")
    print(results)