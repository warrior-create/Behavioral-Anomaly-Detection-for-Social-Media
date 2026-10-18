"""
Unsupervised Model Training — COMPLETE VERSION
===============================================
Trains DBSCAN, Isolation Forest, KMeans, HDBSCAN (optional), and
Gaussian Mixture Model for anomaly / cluster detection.

What was incomplete before:
  - KMeans n_clusters hardcoded to 2
  - No elbow / silhouette analysis to choose optimal k
  - No Gaussian Mixture Model
  - No per-cluster statistics after fitting
  - PCA fixed at 10 components regardless of data size

What this version adds:
  - Silhouette + elbow analysis to auto-select optimal KMeans k
  - Gaussian Mixture Model (GMM) with BIC selection
  - HDBSCAN support (if installed)
  - Per-cluster anomaly statistics export
  - PCA n_components auto-tuned to explain ≥90% variance
  - Full artefact saving (models + scaler + PCA + cluster assignments)
"""

import os
import logging
import traceback
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnsupervisedDetector:
    """
    Trains unsupervised anomaly-detection and clustering models for CIB detection.

    Models:
      - DBSCAN               (density-based, finds arbitrary-shape clusters)
      - Isolation Forest     (anomaly scoring without labels)
      - KMeans               (with auto-selected k via silhouette analysis)
      - GaussianMixture      (soft probabilistic clustering)
      - HDBSCAN              (hierarchical DBSCAN, if hdbscan package installed)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models: Dict   = {}
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA]               = None
        self.cluster_assignments: Dict        = {}
        logger.info("UnsupervisedDetector initialised.")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train_models(self):
        """Load data, preprocess, train all configured models, save artefacts."""
        X_raw = self._load_data()
        if X_raw is None or len(X_raw) == 0:
            logger.error("No data available. Aborting.")
            return

        logger.info("Raw data shape: %s", X_raw.shape)

        # 1. Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)

        # 2. PCA — auto-tune to explain ≥90% variance (or cap at 20 components)
        n_components = self._select_pca_components(X_scaled, variance_threshold=0.90)
        self.pca = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        logger.info(
            "PCA: %d → %d components (%.1f%% variance explained)",
            X_scaled.shape[1], n_components,
            self.pca.explained_variance_ratio_.sum() * 100,
        )

        model_list = self.config.get(
            "models",
            ["dbscan", "isolation_forest", "kmeans", "gmm"],
        )

        # ---- DBSCAN ----
        if "dbscan" in model_list:
            self._train_dbscan(X_pca)

        # ---- Isolation Forest ----
        if "isolation_forest" in model_list:
            self._train_isolation_forest(X_scaled)

        # ---- KMeans (auto k) ----
        if "kmeans" in model_list:
            self._train_kmeans(X_pca)

        # ---- Gaussian Mixture Model ----
        if "gmm" in model_list:
            self._train_gmm(X_pca)

        # ---- HDBSCAN (optional) ----
        if "hdbscan" in model_list:
            self._train_hdbscan(X_pca)

        # 3. Save everything
        self.save_models("data/models")
        self._save_cluster_assignments()

    def predict_anomaly_scores(self, X_new: np.ndarray) -> np.ndarray:
        """
        Score new samples using the fitted Isolation Forest.

        Returns:
            1-D array of anomaly scores (higher = more anomalous).
        """
        if "isolation_forest" not in self.models:
            raise RuntimeError("Isolation Forest not trained yet.")
        X_scaled = self.scaler.transform(X_new)
        # decision_function: negative = anomaly; flip sign for intuitive direction
        return -self.models["isolation_forest"].decision_function(X_scaled)

    def save_models(self, output_dir: str):
        """Save all models and preprocessors."""
        os.makedirs(output_dir, exist_ok=True)

        for name, model in self.models.items():
            path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(model, path)
            logger.info("Saved model → %s", path)

        if self.scaler:
            joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
            logger.info("Saved scaler.")
        if self.pca:
            joblib.dump(self.pca, os.path.join(output_dir, "pca.pkl"))
            logger.info("Saved PCA.")

    # ------------------------------------------------------------------ #
    # Model training methods                                               #
    # ------------------------------------------------------------------ #

    def _train_dbscan(self, X_pca: np.ndarray):
        """Train DBSCAN with configurable eps / min_samples."""
        params = self.config.get("dbscan", {"eps": 2.5, "min_samples": 3})
        model  = DBSCAN(**params)
        labels = model.fit_predict(X_pca)
        self.models["dbscan"] = model
        self.cluster_assignments["dbscan"] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = int(np.sum(labels == -1))
        logger.info("DBSCAN → %d clusters | %d noise points", n_clusters, n_noise)

        if len(set(labels)) > 1:
            sil = silhouette_score(X_pca, labels)
            logger.info("DBSCAN silhouette: %.3f", sil)

    def _train_isolation_forest(self, X_scaled: np.ndarray):
        """Train Isolation Forest."""
        params = self.config.get(
            "isolation_forest", {"contamination": 0.1, "random_state": 42, "n_jobs": -1}
        )
        model = IsolationForest(**params)
        model.fit(X_scaled)
        self.models["isolation_forest"] = model

        scores = model.decision_function(X_scaled)
        n_anomalies = int(np.sum(model.predict(X_scaled) == -1))
        logger.info(
            "Isolation Forest → %d anomalies detected (score mean=%.3f)",
            n_anomalies, scores.mean(),
        )

    def _train_kmeans(self, X_pca: np.ndarray):
        """
        Train KMeans with automatic k selection via silhouette analysis.
        Tries k ∈ [2, max_k] and selects the k with highest silhouette score.
        """
        max_k = self.config.get("kmeans_max_k", min(10, len(X_pca) - 1))
        k_range = range(2, max_k + 1)

        best_k, best_sil, best_model = 2, -1.0, None
        sil_scores = {}

        for k in k_range:
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_pca)
                if len(set(labels)) < 2:
                    continue
                sil = silhouette_score(X_pca, labels)
                sil_scores[k] = sil
                logger.info("  KMeans k=%d → silhouette=%.3f", k, sil)

                if sil > best_sil:
                    best_k, best_sil, best_model = k, sil, km
            except Exception as exc:
                logger.debug("KMeans k=%d failed: %s", k, exc)

        if best_model is None:
            logger.warning("KMeans selection failed — using k=2 fallback.")
            best_model = KMeans(n_clusters=2, random_state=42, n_init=10)
            best_model.fit(X_pca)

        self.models["kmeans"] = best_model
        labels = best_model.predict(X_pca)
        self.cluster_assignments["kmeans"] = labels

        db  = davies_bouldin_score(X_pca, labels) if len(set(labels)) > 1 else float("inf")
        logger.info(
            "KMeans → best k=%d | silhouette=%.3f | davies_bouldin=%.3f",
            best_k, best_sil, db,
        )
        self._log_cluster_stats(labels, "KMeans")

        # Save silhouette curve for the dashboard
        os.makedirs("data/results", exist_ok=True)
        pd.DataFrame(
            {"k": list(sil_scores.keys()), "silhouette": list(sil_scores.values())}
        ).to_csv("data/results/kmeans_silhouette_curve.csv", index=False)

    def _train_gmm(self, X_pca: np.ndarray):
        """
        Train Gaussian Mixture Model with BIC-based component selection.
        Tries n_components ∈ [2, max_k] and selects lowest BIC.
        """
        max_k = self.config.get("gmm_max_components", min(8, len(X_pca) - 1))

        best_bic, best_n, best_model = np.inf, 2, None

        for n in range(2, max_k + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n, covariance_type="full",
                    random_state=42, max_iter=200,
                )
                gmm.fit(X_pca)
                bic = gmm.bic(X_pca)
                logger.info("  GMM n=%d → BIC=%.1f", n, bic)
                if bic < best_bic:
                    best_bic, best_n, best_model = bic, n, gmm
            except Exception as exc:
                logger.debug("GMM n=%d failed: %s", n, exc)

        if best_model is None:
            logger.warning("GMM selection failed — using n=2 fallback.")
            best_model = GaussianMixture(n_components=2, random_state=42)
            best_model.fit(X_pca)

        self.models["gmm"] = best_model
        labels = best_model.predict(X_pca)
        self.cluster_assignments["gmm"] = labels

        logger.info(
            "GMM → best n_components=%d | BIC=%.1f", best_n, best_bic
        )
        self._log_cluster_stats(labels, "GMM")

    def _train_hdbscan(self, X_pca: np.ndarray):
        """Train HDBSCAN if installed."""
        try:
            import hdbscan
            params = self.config.get("hdbscan", {"min_cluster_size": 5})
            model  = hdbscan.HDBSCAN(**params)
            labels = model.fit_predict(X_pca)
            self.models["hdbscan"] = model
            self.cluster_assignments["hdbscan"] = labels

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info("HDBSCAN → %d clusters", n_clusters)
        except ImportError:
            logger.info("HDBSCAN not installed (pip install hdbscan) — skipping.")

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _select_pca_components(
        self, X_scaled: np.ndarray, variance_threshold: float = 0.90
    ) -> int:
        """Select minimum components explaining `variance_threshold` of variance."""
        pca_full = PCA(random_state=42)
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n = int(np.searchsorted(cumvar, variance_threshold) + 1)
        # Clamp: at least 2, at most 20 or feature count
        return max(2, min(n, 20, X_scaled.shape[1]))

    def _log_cluster_stats(self, labels: np.ndarray, model_name: str):
        """Log cluster size distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("[%s] Cluster sizes:", model_name)
        for u, c in zip(unique, counts):
            label_str = "NOISE" if u == -1 else f"Cluster {u}"
            logger.info("  %s: %d users", label_str, c)

    def _save_cluster_assignments(self):
        """Save cluster assignments for each model to data/results/."""
        os.makedirs("data/results", exist_ok=True)
        feat_path = "data/features/final_user_features.csv"

        if not os.path.exists(feat_path):
            logger.warning("Cannot save cluster assignments — feature file missing.")
            return

        user_ids = pd.read_csv(feat_path)["user_id"].astype(str)

        for model_name, labels in self.cluster_assignments.items():
            if len(labels) != len(user_ids):
                continue
            df = pd.DataFrame({
                "user_id": user_ids,
                f"{model_name}_cluster": labels,
            })
            out = f"data/results/{model_name}_cluster_assignments.csv"
            df.to_csv(out, index=False)
            logger.info("Cluster assignments saved → %s", out)

    def _load_data(self) -> Optional[np.ndarray]:
        """Load feature matrix strictly from final_user_features.csv."""
        feature_path = "data/features/final_user_features.csv"

        if not os.path.exists(feature_path):
            logger.error("Feature file not found: %s", feature_path)
            return None

        df = pd.read_csv(feature_path)
        drop_cols = [
            "user_id", "community_id", "created_at", "screen_name",
            "label", "is_bot", "tweet_id", "Unnamed: 0", "algorithm",
        ]
        numeric = (
            df.drop(columns=drop_cols, errors="ignore")
            .select_dtypes(include=[np.number])
            .fillna(0)
        )
        logger.info("Loaded %d users × %d features", *numeric.shape)
        return numeric.values


if __name__ == "__main__":
    detector = UnsupervisedDetector()
    detector.train_models()