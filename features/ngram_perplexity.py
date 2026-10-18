"""
N-gram Language Model & Perplexity Scoring for AI-Generated Text Detection
===========================================================================
Motivation
----------
Bots and AI-generated content often lack the naturalness of human writing.
A character- or word-level n-gram language model trained on *human* text
will assign **high perplexity** to AI/bot-generated text because the
probability of the observed token sequences is low under a human-writing
prior.

This module:
  1. Builds a smoothed n-gram language model (KN-smoothing via NLTK or
     a fast numpy fallback).
  2. Scores each user post's perplexity under the fitted model.
  3. Aggregates per-user statistics (mean, std, max perplexity) as
     ML-ready features.
  4. Provides a flag: "likely_ai_generated" when average perplexity
     exceeds a configurable threshold.

Industry Relevance & Applications
-----------------------------------
  Perplexity-based detection is a core technique in the GenAI safety and
  content-moderation space (e.g. GPTZero-style detectors). Building a
  from-scratch perplexity scorer demonstrates solid statistical NLP theory
  and robust ML engineering capability.
"""

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NgramLanguageModel:
    """
    Smoothed n-gram language model using Laplace (add-k) smoothing.

    Attributes:
        n (int): n-gram order (e.g. 2 = bigram, 3 = trigram).
        k (float): Laplace smoothing constant.
        vocab (set): Observed vocabulary.
        ngram_counts (Counter): Counts of each n-gram.
        context_counts (Counter): Counts of each (n-1)-gram context.
    """

    def __init__(self, n: int = 3, k: float = 0.1):
        """
        Initialise n-gram LM.

        Args:
            n: N-gram order.
            k: Laplace smoothing additive constant (0 < k ≤ 1).
        """
        self.n = n
        self.k = k
        self.vocab: set = set()
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.fitted = False

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokeniser, lowercased."""
        return text.lower().split()

    def fit(self, texts: List[str]):
        """
        Fit the language model on a corpus of texts.

        Args:
            texts: List of documents (training corpus).
        """
        logger.info("NgramLM: fitting %d-gram model on %d documents …", self.n, len(texts))

        self.ngram_counts.clear()
        self.context_counts.clear()
        self.vocab.clear()

        for text in texts:
            tokens = self.tokenize(text)
            self.vocab.update(tokens)

            # Add sentence boundary tokens
            padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]

            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i: i + self.n])
                context = ngram[:-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self.fitted = True
        logger.info(
            "NgramLM fitted | vocab=%d | unique %d-grams=%d",
            len(self.vocab), self.n, len(self.ngram_counts),
        )

    def log_prob(self, ngram: Tuple[str, ...]) -> float:
        """
        Compute log₂ probability of a single n-gram with Laplace smoothing.

        Args:
            ngram: Tuple of n tokens.

        Returns:
            Log₂ probability (negative float; more negative = less likely).
        """
        context = ngram[:-1]
        numerator = self.ngram_counts[ngram] + self.k
        denominator = self.context_counts[context] + self.k * len(self.vocab)
        return math.log2(numerator / denominator)

    def sentence_log_prob(self, text: str) -> float:
        """
        Compute the total log₂ probability of a text under this model.

        Args:
            text: Input text string.

        Returns:
            Sum of log₂ probabilities of all n-grams.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before scoring.")

        tokens = self.tokenize(text)
        if not tokens:
            return 0.0

        padded = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob = 0.0

        for i in range(len(padded) - self.n + 1):
            ngram = tuple(padded[i: i + self.n])
            log_prob += self.log_prob(ngram)

        return log_prob

    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of a text under this language model.

        Perplexity = 2^(-average log₂ probability per token).
        Higher perplexity → the model is *surprised* by the text → likely
        unnatural/bot/AI content.

        Args:
            text: Input text string.

        Returns:
            Perplexity score (≥ 1).
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before scoring.")

        tokens = self.tokenize(text)
        if len(tokens) == 0:
            return float("inf")

        N = len(tokens)
        log_prob = self.sentence_log_prob(text)
        return 2 ** (-log_prob / N)


class PerplexityFeatureExtractor:
    """
    Extracts perplexity-based features for AI-generated / bot text detection.

    Workflow:
      1. The model is fitted on the *majority class* (assumed human) texts,
         or on a separate human reference corpus if provided.
      2. Each user's posts are scored for perplexity.
      3. Per-user aggregate statistics become ML features.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise perplexity extractor.

        Args:
            config: Optional dict with keys:
                - ngram_order (int): n-gram order, default 3
                - smoothing_k (float): Laplace k, default 0.1
                - high_perplexity_threshold (float): flag threshold, default 500
                - reference_corpus_path (str): optional path to a separate
                  human-text CSV for fitting the LM
        """
        self.config = config or {}
        ngram_order = self.config.get("ngram_order", 3)
        smoothing_k = self.config.get("smoothing_k", 0.1)
        self.lm = NgramLanguageModel(n=ngram_order, k=smoothing_k)
        self.high_perplexity_threshold = self.config.get(
            "high_perplexity_threshold", 500.0
        )
        self.reference_corpus_path = self.config.get("reference_corpus_path", None)

        logger.info(
            "PerplexityExtractor | ngram=%d | k=%.2f | threshold=%.1f",
            ngram_order, smoothing_k, self.high_perplexity_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Read a cleaned CSV and produce per-user perplexity features.

        If a reference corpus is configured, the LM is trained on it.
        Otherwise, the LM is trained on all texts in the input file
        (self-referential, which still separates bot from human by
        relative perplexity).

        Args:
            file_path: CSV with 'user_id' and 'text' columns.

        Returns:
            DataFrame with one row per user.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info("Perplexity: loaded %d rows from %s", len(df), file_path)
            return self._extract_from_dataframe(df)
        except Exception as exc:
            logger.error("Perplexity extraction failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    def fit_on_reference(self, reference_texts: List[str]):
        """Fit the LM on an external reference (e.g., known human posts)."""
        self.lm.fit(reference_texts)

    def score_text(self, text: str) -> float:
        """Return perplexity of a single text (LM must be fitted first)."""
        return self.lm.perplexity(text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns:
            logger.error("DataFrame missing 'text' column.")
            return pd.DataFrame()

        if "user_id" not in df.columns:
            df = df.copy()
            df["user_id"] = "all_users"

        all_texts = df["text"].fillna("").astype(str).tolist()

        # Fit the LM
        if self.reference_corpus_path:
            ref_df = pd.read_csv(self.reference_corpus_path)
            ref_texts = ref_df.get("text", pd.Series([])).fillna("").astype(str).tolist()
            logger.info("Fitting LM on %d reference texts", len(ref_texts))
            self.lm.fit(ref_texts)
        else:
            logger.info("Fitting LM on input corpus (%d texts)", len(all_texts))
            self.lm.fit(all_texts)

        # Score each post
        df = df.copy()
        df["perplexity"] = df["text"].apply(
            lambda t: self.lm.perplexity(str(t)) if t else np.nan
        )

        # Aggregate per user
        user_groups = df.groupby("user_id")["perplexity"]
        features = pd.DataFrame(
            {
                "user_id": user_groups.mean().index,
                "perplexity_mean": user_groups.mean().values,
                "perplexity_std": user_groups.std().fillna(0).values,
                "perplexity_min": user_groups.min().values,
                "perplexity_max": user_groups.max().values,
                "perplexity_median": user_groups.median().values,
                "perplexity_q75": user_groups.quantile(0.75).values,
            }
        ).reset_index(drop=True)

        # Derived signal features
        features["perplexity_range"] = (
            features["perplexity_max"] - features["perplexity_min"]
        )
        features["perplexity_cv"] = (
            features["perplexity_std"] / (features["perplexity_mean"] + 1e-9)
        )  # coefficient of variation; low = consistent (bot signal)

        features["perplexity_high_flag"] = (
            features["perplexity_mean"] > self.high_perplexity_threshold
        ).astype(int)

        # Relative perplexity z-score across users
        mean_all = features["perplexity_mean"].mean()
        std_all = features["perplexity_mean"].std() + 1e-9
        features["perplexity_zscore"] = (
            (features["perplexity_mean"] - mean_all) / std_all
        )

        logger.info(
            "Perplexity features computed for %d users | mean=%.1f | std=%.1f",
            len(features),
            features["perplexity_mean"].mean(),
            features["perplexity_mean"].std(),
        )
        return features


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "user_id": [
                "bot_1", "bot_1", "bot_1",
                "bot_2", "bot_2",
                "human_1", "human_1", "human_1",
            ],
            "text": [
                # Repetitive, unnatural bot text
                "win free bitcoin giveaway click now free crypto prize win free",
                "free crypto win bitcoin giveaway click now prize win free bitcoin",
                "bitcoin win giveaway crypto free prize click win now free bitcoin",
                # Another bot
                "urgent breaking news market crash imminent sell everything now",
                "breaking urgent market crash sell everything now imminent",
                # Natural human text
                "I spent the afternoon reading about the history of the Roman Empire.",
                "The documentary about deep sea creatures was absolutely fascinating.",
                "Had a wonderful dinner with friends tonight. Tried a new Thai restaurant.",
            ],
        }
    )

    extractor = PerplexityFeatureExtractor(
        config={"ngram_order": 2, "high_perplexity_threshold": 200}
    )
    features = extractor._extract_from_dataframe(sample_data)
    print("\n=== Perplexity Features ===")
    print(features.to_string(index=False))
