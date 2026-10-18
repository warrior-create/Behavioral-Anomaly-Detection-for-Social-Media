"""
TF-IDF Feature Extraction & Content Similarity Module
Uses TF-IDF vectorization to detect coordinated inauthentic behavior through
content similarity, vocabulary patterns, and bot-like repetition signals.

Key capabilities:
  - TF-IDF vectorization of post corpora
  - Pairwise cosine similarity to surface highly similar (copy-paste) accounts
  - Per-user TF-IDF statistical features for ML training
  - Coordinated user-pair detection based on shared vocabulary
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


class TFIDFAnalyzer:
    """
    Extracts TF-IDF based features to detect coordinated inauthentic behavior.

    Rationale: Bot accounts often recycle the same content verbatim or with
    minor edits. TF-IDF cosine similarity is a classic, explainable signal that
    complements BERT embeddings — BERT captures semantic similarity while TF-IDF
    captures lexical/surface overlap, catching copy-paste bots that BERT may
    normalise away.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise the TF-IDF analyser.

        Args:
            config: Optional configuration dict. Supported keys:
                - max_features (int): vocabulary cap, default 10000
                - ngram_range (tuple): e.g. (1, 2) for unigrams+bigrams
                - similarity_threshold (float): threshold for flagging
                  coordinated pairs, default 0.8
                - min_df (int|float): minimum document frequency
        """
        self.config = config or {}
        self.max_features = self.config.get("max_features", 10_000)
        self.ngram_range = tuple(self.config.get("ngram_range", [1, 2]))
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.min_df = self.config.get("min_df", 2)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,          # log(1+tf) — dampens high-freq terms
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )

        self.tfidf_matrix = None        # (n_posts, vocab_size) sparse matrix
        self.feature_names = None       # vocabulary list
        self.fitted = False

        logger.info(
            "TFIDFAnalyzer initialised | max_features=%d | ngram=%s | sim_thresh=%.2f",
            self.max_features, self.ngram_range, self.similarity_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Main entry point: read a cleaned CSV and return per-user TF-IDF features.

        Args:
            file_path: Path to CSV with at minimum columns 'user_id' and 'text'.

        Returns:
            DataFrame with one row per user and TF-IDF derived numerical features.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info("TF-IDF: loaded %d rows from %s", len(df), file_path)

            if "text" not in df.columns:
                logger.error("No 'text' column found in %s", file_path)
                return pd.DataFrame()

            # Aggregate all posts per user into a single document
            user_docs = self._aggregate_user_docs(df)

            # Fit & transform
            self.tfidf_matrix = self.vectorizer.fit_transform(user_docs["agg_text"])
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.fitted = True
            logger.info("TF-IDF: vocab size = %d", len(self.feature_names))

            # Build per-user numeric features
            user_features = self._compute_user_features(user_docs)

            # Compute pairwise similarity and add coordination flag
            user_features = self._add_coordination_features(user_features, user_docs)

            logger.info("TF-IDF features computed for %d users", len(user_features))
            return user_features

        except Exception as exc:
            logger.error("TF-IDF feature extraction failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizer on a list of texts and return dense TF-IDF matrix.

        Args:
            texts: List of documents.

        Returns:
            Dense numpy array of shape (n_docs, vocab_size).
        """
        matrix = self.vectorizer.fit_transform(texts)
        self.tfidf_matrix = matrix
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.fitted = True
        return matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using the fitted vectorizer."""
        if not self.fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        return self.vectorizer.transform(texts).toarray()

    def compute_pairwise_similarity(
        self, matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute full pairwise cosine similarity matrix.

        Args:
            matrix: Optional dense/sparse array. Defaults to fitted tfidf_matrix.

        Returns:
            (n, n) similarity matrix.
        """
        mat = matrix if matrix is not None else self.tfidf_matrix
        if mat is None:
            raise RuntimeError("No TF-IDF matrix available. Run extract_features first.")
        sim = cosine_similarity(mat)
        logger.info("Pairwise similarity matrix computed: %s", sim.shape)
        return sim

    def find_coordinated_pairs(
        self,
        user_ids: List[str],
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of users with suspiciously high content similarity.

        Args:
            user_ids: Ordered list matching rows of the fitted tfidf_matrix.
            threshold: Override default similarity_threshold.

        Returns:
            List of (user_a, user_b, similarity_score) tuples above threshold.
        """
        thresh = threshold if threshold is not None else self.similarity_threshold
        sim_matrix = self.compute_pairwise_similarity()
        pairs = []

        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                score = float(sim_matrix[i, j])
                if score >= thresh:
                    pairs.append((user_ids[i], user_ids[j], score))

        pairs.sort(key=lambda x: -x[2])
        logger.info(
            "Found %d coordinated pairs at threshold=%.2f", len(pairs), thresh
        )
        return pairs

    def get_top_terms(self, user_doc_index: int, top_n: int = 15) -> List[Tuple[str, float]]:
        """
        Return the top TF-IDF terms for a given document (user-aggregate).

        Args:
            user_doc_index: Row index in the fitted tfidf_matrix.
            top_n: Number of top terms to return.

        Returns:
            List of (term, tfidf_score) sorted descending.
        """
        if not self.fitted:
            raise RuntimeError("Vectorizer not fitted yet.")

        row = self.tfidf_matrix[user_doc_index]
        if issparse(row):
            row = row.toarray().flatten()

        top_indices = np.argsort(row)[::-1][:top_n]
        return [(self.feature_names[i], float(row[i])) for i in top_indices if row[i] > 0]

    def save_matrix(self, output_path: str):
        """Persist fitted TF-IDF matrix as a .npz file."""
        if self.tfidf_matrix is None:
            logger.warning("No matrix to save.")
            return
        from scipy.sparse import save_npz
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_npz(output_path, self.tfidf_matrix)
        logger.info("TF-IDF matrix saved to %s", output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_user_docs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Concatenate all posts per user into a single document."""
        if "user_id" not in df.columns:
            df["user_id"] = "all_users"

        user_docs = (
            df.groupby("user_id")["text"]
            .apply(lambda posts: " ".join(posts.fillna("").astype(str)))
            .reset_index()
            .rename(columns={"text": "agg_text"})
        )
        user_docs["post_count"] = df.groupby("user_id").size().values
        return user_docs

    def _compute_user_features(self, user_docs: pd.DataFrame) -> pd.DataFrame:
        """
        Derive interpretable numerical features from TF-IDF vectors.

        Features:
            tfidf_mean       : average TF-IDF weight across all vocab terms
            tfidf_max        : highest single-term TF-IDF weight
            tfidf_l2_norm    : L2 norm of the TF-IDF vector
            tfidf_nnz        : number of non-zero terms (vocabulary breadth)
            tfidf_nnz_ratio  : vocabulary breadth as fraction of max_features
            tfidf_entropy    : Shannon entropy of TF-IDF weights
                               (low = dominated by few keywords → bot signal)
            tfidf_gini       : Gini coefficient of term weights
                               (high = unequal distribution → repetitive content)
        """
        records = []
        n_users = self.tfidf_matrix.shape[0]

        for i in range(n_users):
            row = self.tfidf_matrix[i]
            if issparse(row):
                row = row.toarray().flatten()

            nnz = int(np.count_nonzero(row))
            nnz_ratio = nnz / max(len(row), 1)
            nonzero_vals = row[row > 0]

            # Shannon entropy
            if len(nonzero_vals) > 0:
                p = nonzero_vals / nonzero_vals.sum()
                entropy = float(-np.sum(p * np.log2(p + 1e-12)))
            else:
                entropy = 0.0

            # Gini coefficient
            gini = self._gini(nonzero_vals) if len(nonzero_vals) > 1 else 0.0

            records.append(
                {
                    "user_id": user_docs.iloc[i]["user_id"],
                    "tfidf_mean": float(np.mean(nonzero_vals)) if len(nonzero_vals) else 0.0,
                    "tfidf_max": float(np.max(nonzero_vals)) if len(nonzero_vals) else 0.0,
                    "tfidf_l2_norm": float(np.linalg.norm(row)),
                    "tfidf_nnz": nnz,
                    "tfidf_nnz_ratio": nnz_ratio,
                    "tfidf_entropy": entropy,
                    "tfidf_gini": gini,
                    "post_count": int(user_docs.iloc[i]["post_count"]),
                }
            )

        return pd.DataFrame(records)

    def _add_coordination_features(
        self, user_features: pd.DataFrame, user_docs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add per-user coordination score: how many other users share very
        similar content with this user.
        """
        sim_matrix = self.compute_pairwise_similarity()
        n = sim_matrix.shape[0]

        coord_counts = []
        avg_sim_scores = []

        for i in range(n):
            others = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i + 1:]])
            coord_count = int(np.sum(others >= self.similarity_threshold))
            avg_sim = float(np.mean(others))
            coord_counts.append(coord_count)
            avg_sim_scores.append(avg_sim)

        user_features = user_features.copy()
        user_features["tfidf_coord_partner_count"] = coord_counts
        user_features["tfidf_avg_similarity_to_others"] = avg_sim_scores
        user_features["tfidf_is_coordinated"] = (
            user_features["tfidf_coord_partner_count"] > 0
        ).astype(int)

        return user_features

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Compute Gini coefficient for a 1-D array of non-negative values."""
        if len(values) == 0:
            return 0.0
        values = np.sort(values)
        n = len(values)
        cum = np.cumsum(values)
        return float((2 * np.sum((np.arange(1, n + 1)) * values) / (n * cum[-1])) - (n + 1) / n)


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    sample_texts = [
        "Win free crypto NOW! Click here to claim your prize giveaway!",
        "win free crypto now! click here claim prize giveaway",  # near-duplicate (bot)
        "Breaking news: market crash imminent, sell everything",
        "BREAKING NEWS market crash imminent sell now",           # near-duplicate (bot)
        "Just had a lovely afternoon walk in the park today.",
        "Python is a great language for data science projects.",
    ]
    sample_users = ["bot_1", "bot_2", "bot_3", "bot_4", "human_1", "human_2"]

    analyser = TFIDFAnalyzer(config={"similarity_threshold": 0.5, "min_df": 1})
    analyser.fit_transform(sample_texts)

    print("\n=== Top terms per user-doc ===")
    for idx, uid in enumerate(sample_users):
        top = analyser.get_top_terms(idx, top_n=5)
        print(f"  {uid}: {top}")

    print("\n=== Coordinated pairs ===")
    pairs = analyser.find_coordinated_pairs(sample_users)
    for u1, u2, score in pairs:
        print(f"  {u1} <-> {u2}  sim={score:.3f}")
