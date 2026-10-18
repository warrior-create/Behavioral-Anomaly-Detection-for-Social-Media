"""
LDA Topic Modelling for CIB Detection
Discovers latent topics in social media corpora to surface coordinated
narrative campaigns pushed by bot networks.

Key capabilities:
  - Fit LDA on a corpus of user posts
  - Assign dominant topic distributions per user
  - Detect topic concentration (bots hammer a single topic)
  - Find users sharing extremely similar topic distributions (coordination)
  - Produce topic keywords for human-readable interpretation

Why LDA is critical:
  - Demonstrates unsupervised learning applied to text
  - Shows understanding of probabilistic generative models
  - Provides interpretable output (topic keywords) stakeholders can read
  - Scales to large corpora without labelled data to find signal in operational data
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    LDA-based topic modelling for coordinated inauthentic behaviour detection.

    The key hypothesis: bot accounts are typically tasked with amplifying a
    narrow set of narratives.  Their per-post (and per-user aggregate) topic
    distributions will therefore be extremely *concentrated* on one or two
    topics, whereas genuine human accounts show broader topic diversity.
    Additionally, coordinated bot clusters will have *similar* topic
    distributions — detectable via Jensen-Shannon divergence.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise topic modeler.

        Args:
            config: Optional dict with keys:
                - n_topics (int): number of LDA topics, default 10
                - max_features (int): vocabulary cap, default 5000
                - max_iter (int): LDA EM iterations, default 20
                - random_state (int): reproducibility seed, default 42
                - top_n_words (int): words per topic summary, default 10
                - coord_jsd_threshold (float): JS-divergence below which two
                  users are flagged as coordinated, default 0.1
        """
        self.config = config or {}
        self.n_topics = self.config.get("n_topics", 10)
        self.max_features = self.config.get("max_features", 5_000)
        self.max_iter = self.config.get("max_iter", 20)
        self.random_state = self.config.get("random_state", 42)
        self.top_n_words = self.config.get("top_n_words", 10)
        self.coord_jsd_threshold = self.config.get("coord_jsd_threshold", 0.10)

        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english",
            min_df=2,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # only alpha tokens ≥3 chars
        )
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.max_iter,
            learning_method="online",
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.term_matrix = None         # (n_docs, vocab)
        self.topic_distributions = None  # (n_docs, n_topics)
        self.fitted = False

        logger.info(
            "TopicModeler initialised | n_topics=%d | max_features=%d",
            self.n_topics, self.max_features,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Main entry point: read CSV and return per-user topic features.

        Args:
            file_path: CSV with 'user_id' and 'text' columns.

        Returns:
            DataFrame with one row per user.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info("TopicModeler: loaded %d rows from %s", len(df), file_path)
            return self._extract_from_dataframe(df)
        except Exception as exc:
            logger.error("Topic modelling failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    def fit(self, texts: List[str]):
        """
        Fit the LDA model on a list of documents.

        Args:
            texts: List of raw/cleaned text strings.
        """
        logger.info("TopicModeler: fitting on %d documents …", len(texts))
        self.term_matrix = self.vectorizer.fit_transform(texts)
        self.topic_distributions = self.lda_model.fit_transform(self.term_matrix)
        self.fitted = True
        logger.info(
            "LDA fitted | perplexity=%.2f", self.lda_model.perplexity(self.term_matrix)
        )

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts to topic distributions (must call fit first)."""
        if not self.fitted:
            raise RuntimeError("Call fit() or extract_features() first.")
        term_mat = self.vectorizer.transform(texts)
        return self.lda_model.transform(term_mat)

    def get_topic_keywords(self) -> Dict[int, List[str]]:
        """
        Return the top keywords for each topic.

        Returns:
            {topic_id: [word1, word2, …]} dict.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")

        vocab = np.array(self.vectorizer.get_feature_names_out())
        keywords = {}
        for topic_idx, topic_vec in enumerate(self.lda_model.components_):
            top_word_indices = topic_vec.argsort()[: -self.top_n_words - 1: -1]
            keywords[topic_idx] = vocab[top_word_indices].tolist()
        return keywords

    def print_topics(self):
        """Print a human-readable summary of all topics."""
        keywords = self.get_topic_keywords()
        print(f"\n{'='*60}")
        print(f"  LDA Topics (n={self.n_topics})")
        print(f"{'='*60}")
        for tid, words in keywords.items():
            print(f"  Topic {tid:2d}: {', '.join(words)}")
        print(f"{'='*60}\n")

    def find_coordinated_users_by_topic(
        self,
        user_ids: List[str],
        topic_dists: Optional[np.ndarray] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find user pairs with very similar topic distributions (low JS-divergence).

        Args:
            user_ids: Ordered list of user IDs matching rows of topic_dists.
            topic_dists: (n_users, n_topics) array. Defaults to fitted distributions.

        Returns:
            List of (user_a, user_b, jsd) tuples below coord_jsd_threshold.
        """
        dists = topic_dists if topic_dists is not None else self.topic_distributions
        if dists is None:
            raise RuntimeError("No topic distributions available.")

        coord_pairs = []
        for i, uid1 in enumerate(user_ids):
            for j, uid2 in enumerate(user_ids):
                if j <= i:
                    continue
                jsd = float(jensenshannon(dists[i], dists[j]))
                if jsd <= self.coord_jsd_threshold:
                    coord_pairs.append((uid1, uid2, jsd))

        coord_pairs.sort(key=lambda x: x[2])
        logger.info(
            "Found %d topic-coordinated pairs (JSD ≤ %.2f)",
            len(coord_pairs), self.coord_jsd_threshold,
        )
        return coord_pairs

    def save_model(self, output_dir: str):
        """Persist vectorizer and LDA model to disk using joblib."""
        import joblib
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, out / "lda_vectorizer.pkl")
        joblib.dump(self.lda_model, out / "lda_model.pkl")
        logger.info("LDA model saved to %s", output_dir)

    def load_model(self, output_dir: str):
        """Load a previously saved vectorizer and LDA model."""
        import joblib
        out = Path(output_dir)
        self.vectorizer = joblib.load(out / "lda_vectorizer.pkl")
        self.lda_model = joblib.load(out / "lda_model.pkl")
        self.fitted = True
        logger.info("LDA model loaded from %s", output_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full extraction pipeline on a DataFrame."""
        if "text" not in df.columns:
            logger.error("DataFrame missing 'text' column.")
            return pd.DataFrame()

        if "user_id" not in df.columns:
            df = df.copy()
            df["user_id"] = "all_users"

        # Aggregate posts per user
        user_docs = (
            df.groupby("user_id")["text"]
            .apply(lambda x: " ".join(x.fillna("").astype(str)))
            .reset_index()
            .rename(columns={"text": "agg_text"})
        )

        self.fit(user_docs["agg_text"].tolist())

        topic_dists = self.topic_distributions    # (n_users, n_topics)
        feature_rows = []

        for i, uid in enumerate(user_docs["user_id"]):
            dist = topic_dists[i]
            dominant_topic = int(np.argmax(dist))
            dominant_weight = float(dist[dominant_topic])

            # Entropy of topic distribution (low → focused on few topics → bot)
            entropy = float(-np.sum(dist * np.log2(dist + 1e-12)))

            # Concentration: max minus mean (high → dominated by one topic)
            concentration = float(np.max(dist) - np.mean(dist))

            row = {"user_id": uid, "lda_dominant_topic": dominant_topic}
            row["lda_dominant_topic_weight"] = dominant_weight
            row["lda_topic_entropy"] = entropy
            row["lda_topic_concentration"] = concentration
            row["lda_topic_gini"] = self._gini(dist)

            # Per-topic weights as individual features
            for t in range(self.n_topics):
                row[f"lda_topic_{t}_weight"] = float(dist[t])

            feature_rows.append(row)

        features_df = pd.DataFrame(feature_rows)

        # Add pairwise coordination signal
        coord_pairs = self.find_coordinated_users_by_topic(
            user_docs["user_id"].tolist(), topic_dists
        )
        coord_counts: Dict[str, int] = defaultdict(int)
        for u1, u2, _ in coord_pairs:
            coord_counts[u1] += 1
            coord_counts[u2] += 1

        features_df["lda_coord_topic_partners"] = features_df["user_id"].map(
            lambda uid: coord_counts.get(uid, 0)
        )

        logger.info("Topic features extracted for %d users", len(features_df))
        return features_df

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Gini coefficient measuring inequality of topic distribution."""
        values = np.sort(np.abs(values))
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0.0
        cum = np.cumsum(values)
        return float((2 * np.sum(np.arange(1, n + 1) * values) / (n * cum[-1])) - (n + 1) / n)


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "user_id": [
                "bot_a", "bot_a", "bot_a",
                "bot_b", "bot_b", "bot_b",
                "human_1", "human_1", "human_1",
                "human_2", "human_2",
            ],
            "text": [
                "Bitcoin giveaway free crypto click win prize now",
                "Free bitcoin giveaway win crypto prize now click here",
                "Claim free crypto giveaway bitcoin prize urgent click",
                "Bitcoin giveaway free crypto click win prize now",
                "Earn free crypto bitcoin giveaway win prizes click",
                "Crypto giveaway free bitcoin win click now prizes",
                "Just finished a really interesting book about history",
                "The weather today is lovely. Went for a long walk.",
                "Cooking is such a relaxing hobby. Tried a new pasta recipe.",
                "Machine learning models are fascinating to train.",
                "Python makes data analysis really approachable.",
            ],
        }
    )

    modeler = TopicModeler(config={"n_topics": 5, "max_features": 200, "min_df": 1})
    features = modeler._extract_from_dataframe(sample_data)
    modeler.print_topics()
    print("=== Per-user topic features (sample columns) ===")
    summary_cols = [
        "user_id", "lda_dominant_topic", "lda_dominant_topic_weight",
        "lda_topic_entropy", "lda_topic_concentration", "lda_coord_topic_partners",
    ]
    print(features[summary_cols].to_string(index=False))
