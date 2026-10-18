"""
PMI Keyword Co-occurrence Matrix for Coordinated Talking-Point Detection
========================================================================
Pointwise Mutual Information (PMI) measures how much more (or less) often
two terms co-occur than expected by chance:

    PMI(x, y) = log[ P(x,y) / (P(x) · P(y)) ]

A high PMI between two keywords signals a *meaningful association* — they
appear together far more than random.

For CIB detection the key insight is:
  Bot networks are briefed with *shared talking points*.  The PMI matrix
  of a bot's vocabulary will show tight, repeated co-occurrence clusters
  around a narrow set of propaganda terms.  Human text, by contrast, shows
  a broad, diffuse co-occurrence graph.

This module:
  1. Builds a term × term PMI matrix from a corpus.
  2. Extracts per-user PMI-based features (vocabulary cohesion,
     top-PMI pairs, bot-term co-occurrence density).
  3. Exposes a coordination signal: two users whose *top PMI pairs* heavily
     overlap are likely co-ordinated.

References
----------
  Church & Hanks (1990) — "Word Association Norms, Mutual Information,
  and Lexicography."
"""

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known bot / propaganda talking-point terms (lowercased)
_PROPAGANDA_TERMS: Set[str] = {
    "giveaway", "free", "win", "prize", "click", "crypto", "bitcoin",
    "breaking", "urgent", "exclusive", "deep state", "fake news",
    "mainstream media", "wake up", "shill", "bot", "banned", "censored",
    "share", "retweet", "spread", "truth", "lie", "scam", "exposed",
}


class PMICooccurrenceAnalyzer:
    """
    Computes a PMI co-occurrence matrix and derives per-user NLP features
    for CIB / bot detection.

    Parameters
    ----------
    window_size : int
        Number of tokens on each side of the target token to consider
        as co-occurring.  Default = 5.
    min_freq : int
        Minimum frequency for a term to be included in the vocabulary.
    top_pairs_per_user : int
        How many top PMI pairs to keep per user for coordination scoring.
    coord_overlap_threshold : float
        Fraction of top PMI pairs shared between two users to flag them
        as coordinated.  Default = 0.5.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.window_size = self.config.get("window_size", 5)
        self.min_freq = self.config.get("min_freq", 2)
        self.top_pairs_per_user = self.config.get("top_pairs_per_user", 20)
        self.coord_overlap_threshold = self.config.get("coord_overlap_threshold", 0.5)

        # Corpus-level statistics (populated by fit())
        self._term_freq: Counter = Counter()
        self._cooccurrence: Counter = Counter()
        self._total_tokens: int = 0
        self._total_pairs: int = 0
        self.vocab: List[str] = []
        self.fitted: bool = False

        logger.info(
            "PMICooccurrenceAnalyzer | window=%d | min_freq=%d",
            self.window_size, self.min_freq,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Main entry point. Read a cleaned CSV and return per-user PMI features.

        Args:
            file_path: CSV with 'user_id' and 'text' columns.

        Returns:
            DataFrame with one row per user.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info("PMI: loaded %d rows from %s", len(df), file_path)
            return self._extract_from_dataframe(df)
        except Exception as exc:
            logger.error("PMI extraction failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    def fit(self, texts: List[str]):
        """
        Build corpus-level term frequency and co-occurrence counts.

        Args:
            texts: List of tokenisable text documents.
        """
        logger.info("PMI: fitting co-occurrence model on %d texts …", len(texts))
        self._term_freq.clear()
        self._cooccurrence.clear()
        self._total_tokens = 0
        self._total_pairs = 0

        for text in texts:
            tokens = self._tokenize(text)
            self._term_freq.update(tokens)
            self._total_tokens += len(tokens)

            # Sliding window co-occurrence
            for i, token in enumerate(tokens):
                window_start = max(0, i - self.window_size)
                window_end = min(len(tokens), i + self.window_size + 1)
                for j in range(window_start, window_end):
                    if j == i:
                        continue
                    pair = tuple(sorted([token, tokens[j]]))
                    self._cooccurrence[pair] += 1
                    self._total_pairs += 1

        # Build vocabulary from min_freq terms
        self.vocab = [
            t for t, c in self._term_freq.items() if c >= self.min_freq
        ]
        self.fitted = True
        logger.info(
            "PMI model fitted | vocab=%d | unique pairs=%d",
            len(self.vocab), len(self._cooccurrence),
        )

    def pmi(self, term_a: str, term_b: str, positive: bool = True) -> float:
        """
        Compute PMI between two terms.

        Args:
            term_a, term_b: Terms to compare.
            positive: If True, clip at 0 (Positive PMI, PPMI).

        Returns:
            PMI score.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        fa = self._term_freq.get(term_a, 0)
        fb = self._term_freq.get(term_b, 0)
        pair = tuple(sorted([term_a, term_b]))
        fab = self._cooccurrence.get(pair, 0)

        if fa == 0 or fb == 0 or fab == 0:
            return 0.0

        pa = fa / self._total_tokens
        pb = fb / self._total_tokens
        pab = fab / self._total_pairs

        score = math.log2(pab / (pa * pb))
        return max(0.0, score) if positive else score

    def get_top_pmi_pairs(
        self, texts: List[str], top_n: int = 20
    ) -> List[Tuple[str, str, float]]:
        """
        Get the top-N highest PPMI pairs within a given set of texts.

        Args:
            texts: Documents to analyse.
            top_n: Number of pairs to return.

        Returns:
            List of (term_a, term_b, pmi_score) sorted descending.
        """
        local_cooccurrence: Counter = Counter()
        local_term_freq: Counter = Counter()
        local_total = 0
        local_pairs = 0

        for text in texts:
            tokens = self._tokenize(text)
            local_term_freq.update(tokens)
            local_total += len(tokens)
            for i, tok in enumerate(tokens):
                ws = max(0, i - self.window_size)
                we = min(len(tokens), i + self.window_size + 1)
                for j in range(ws, we):
                    if j == i:
                        continue
                    pair = tuple(sorted([tok, tokens[j]]))
                    local_cooccurrence[pair] += 1
                    local_pairs += 1

        pairs_scored = []
        for pair, fab in local_cooccurrence.items():
            fa = local_term_freq.get(pair[0], 0)
            fb = local_term_freq.get(pair[1], 0)
            if fa < 2 or fb < 2 or fab < 2:
                continue
            pa = fa / max(local_total, 1)
            pb = fb / max(local_total, 1)
            pab = fab / max(local_pairs, 1)
            score = math.log2(pab / (pa * pb + 1e-12))
            if score > 0:
                pairs_scored.append((pair[0], pair[1], score))

        pairs_scored.sort(key=lambda x: -x[2])
        return pairs_scored[:top_n]

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
        self.fit(all_texts)

        # Per-user feature extraction
        user_records: Dict[str, Dict] = {}
        user_top_pairs: Dict[str, Set[Tuple[str, str]]] = {}

        for uid, group in df.groupby("user_id"):
            texts = group["text"].fillna("").astype(str).tolist()
            tokens_all = " ".join(texts).split()

            # 1. Vocabulary richness in terms of PPMI coverage
            user_token_set = set(self._tokenize(" ".join(texts)))
            vocab_coverage = len(user_token_set & set(self.vocab)) / max(len(self.vocab), 1)

            # 2. Top PPMI pairs for this user
            top_pairs = self.get_top_pmi_pairs(texts, top_n=self.top_pairs_per_user)
            top_pair_set = {(a, b) for a, b, _ in top_pairs}
            user_top_pairs[str(uid)] = top_pair_set

            # 3. Average PPMI of user's top pairs
            avg_top_pmi = float(np.mean([s for _, _, s in top_pairs])) if top_pairs else 0.0

            # 4. Propaganda / bot-term co-occurrence density
            prop_hits = sum(
                1 for (a, b) in top_pair_set
                if a in _PROPAGANDA_TERMS or b in _PROPAGANDA_TERMS
            )
            prop_density = prop_hits / max(len(top_pair_set), 1)

            # 5. PMI concentration (std of PMI scores; low = uniform bot vocab)
            pmi_scores = [s for _, _, s in top_pairs]
            pmi_std = float(np.std(pmi_scores)) if len(pmi_scores) > 1 else 0.0

            user_records[str(uid)] = {
                "user_id": str(uid),
                "pmi_vocab_coverage": vocab_coverage,
                "pmi_avg_top_pmi": avg_top_pmi,
                "pmi_propaganda_density": prop_density,
                "pmi_top_pair_count": len(top_pair_set),
                "pmi_score_std": pmi_std,
                "pmi_unique_tokens": len(user_token_set),
            }

        # Coordination: fraction of top PMI pairs shared between user pairs
        user_ids = list(user_records.keys())
        coord_scores: Dict[str, float] = defaultdict(float)
        coord_counts: Dict[str, int] = defaultdict(int)

        for i, u1 in enumerate(user_ids):
            for u2 in user_ids[i + 1:]:
                set1 = user_top_pairs[u1]
                set2 = user_top_pairs[u2]
                union = set1 | set2
                intersection = set1 & set2
                jaccard = len(intersection) / len(union) if union else 0.0

                if jaccard >= self.coord_overlap_threshold:
                    coord_counts[u1] += 1
                    coord_counts[u2] += 1
                coord_scores[u1] += jaccard
                coord_scores[u2] += jaccard

        for uid in user_ids:
            n_others = max(len(user_ids) - 1, 1)
            user_records[uid]["pmi_coord_partner_count"] = coord_counts.get(uid, 0)
            user_records[uid]["pmi_avg_pair_overlap"] = coord_scores.get(uid, 0.0) / n_others

        features_df = pd.DataFrame(list(user_records.values()))
        logger.info("PMI features extracted for %d users", len(features_df))
        return features_df

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase whitespace tokeniser filtering short/stop tokens."""
        import re
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        _STOPWORDS = {
            "the", "and", "for", "this", "that", "with", "from", "are",
            "was", "not", "but", "have", "has", "you", "your", "our",
        }
        return [t for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "user_id": ["bot", "bot", "bot", "human", "human", "human"],
            "text": [
                "Win free bitcoin giveaway click now crypto prize win",
                "Free bitcoin giveaway crypto prize click win now free",
                "Click now win free crypto bitcoin giveaway prize urgent",
                "I enjoy reading science fiction novels on weekends.",
                "The weather was perfect for a long hike in the mountains.",
                "Machine learning continues to transform the software industry.",
            ],
        }
    )

    analyser = PMICooccurrenceAnalyzer(
        config={"window_size": 3, "min_freq": 1, "top_pairs_per_user": 10}
    )
    features = analyser._extract_from_dataframe(sample_data)
    print("\n=== PMI Features ===")
    print(features.to_string(index=False))
