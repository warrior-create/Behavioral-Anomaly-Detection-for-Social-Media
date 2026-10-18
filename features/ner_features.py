"""
Named Entity Recognition (NER) Feature Extraction
Uses spaCy to identify entity-level coordination signals in social media text.

Key insight for CIB detection:
  Bot networks often amplify the *same* set of entities (organisations, people,
  locations, products) in concert.  By extracting entity distributions per user
  and comparing them across users, we get a strong coordination signal that is
  complementary to both stylometry and TF-IDF similarity.

Features produced (per user):
  - Absolute and relative entity-type counts (PERSON, ORG, GPE, …)
  - Entity diversity (unique entity ratio)
  - Top-entity overlap score between every pair of users (coordination signal)
  - NER-based "bot vocabulary" score: fraction of entities on a watchlist
"""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# spaCy is loaded lazily so the module can be imported even without a model
_NLP = None


def _get_nlp(model: str = "en_core_web_sm"):
    """Load spaCy model once and cache it."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load(model)
            logger.info("spaCy model loaded: %s", model)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Run:  python -m spacy download %s",
                model, model,
            )
            raise
    return _NLP


# Entities that are disproportionately common in bot / misinformation campaigns
_BOT_ENTITY_WATCHLIST = frozenset({
    "crypto", "bitcoin", "ethereum", "giveaway", "free", "click", "win",
    "breaking", "urgent", "exclusive", "cia", "fbi", "nwo", "deep state",
    "illuminati", "qanon", "antifa", "soros", "rothschild",
})


class NERFeatureExtractor:
    """
    Extracts Named Entity Recognition features for CIB detection.

    Attributes:
        spacy_model: Name of the spaCy model to use.
        entity_types: Which NE types to track (None = all).
        batch_size: spaCy pipe batch size for throughput.
        watchlist: Frozenset of bot-adjacent entity strings (lowercased).
    """

    # Entity types we care about most for social media CIB
    DEFAULT_ENTITY_TYPES = {
        "PERSON", "ORG", "GPE",   # geo-political entity
        "LOC", "PRODUCT", "EVENT",
        "NORP",                    # nationalities, religions, political groups
        "MONEY", "LAW",
    }

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise NER extractor.

        Args:
            config: Optional dict with keys:
                - spacy_model (str): spaCy model name, default 'en_core_web_sm'
                - entity_types (list): entity type filter
                - batch_size (int): spaCy pipe batch size
                - watchlist (list): additional bot-signal entity strings
        """
        self.config = config or {}
        self.spacy_model = self.config.get("spacy_model", "en_core_web_sm")
        self.entity_types = set(
            self.config.get("entity_types", list(self.DEFAULT_ENTITY_TYPES))
        )
        self.batch_size = self.config.get("batch_size", 64)
        extra_watchlist = set(
            w.lower() for w in self.config.get("watchlist", [])
        )
        self.watchlist = _BOT_ENTITY_WATCHLIST | extra_watchlist

        logger.info(
            "NERFeatureExtractor initialised | model=%s | entity_types=%s",
            self.spacy_model, sorted(self.entity_types),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Read a cleaned CSV and return per-user NER feature DataFrame.

        Args:
            file_path: CSV with columns 'user_id' and 'text'.

        Returns:
            DataFrame with one row per user.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info("NER: loaded %d rows from %s", len(df), file_path)
            return self._extract_from_dataframe(df)
        except Exception as exc:
            logger.error("NER feature extraction failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Public wrapper around internal method (useful for testing)."""
        return self._extract_from_dataframe(df)

    def extract_entities_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from a single piece of text.

        Args:
            text: Raw or cleaned text string.

        Returns:
            List of (entity_text, entity_label) tuples.
        """
        nlp = _get_nlp(self.spacy_model)
        doc = nlp(text[:10_000])  # cap to avoid memory issues on huge texts
        return [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.label_ in self.entity_types
        ]

    def compute_entity_overlap(
        self, user_entity_sets: Dict[str, Counter]
    ) -> pd.DataFrame:
        """
        Compute Jaccard similarity of entity sets efficiently using an inverted index.
        Only candidate user pairs sharing at least one entity are evaluated.

        Args:
            user_entity_sets: {user_id: Counter of entities}.

        Returns:
            DataFrame with columns ['user_a', 'user_b', 'entity_jaccard'].
        """
        active_users = {
            u: set(c.keys()) for u, c in user_entity_sets.items() if len(c) > 0
        }
        if len(active_users) < 2:
            return pd.DataFrame(columns=["user_a", "user_b", "entity_jaccard"])

        # Inverted index: entity -> set of user_ids
        ent_to_users: Dict[str, set] = defaultdict(set)
        for u, ents in active_users.items():
            for ent in ents:
                ent_to_users[ent].add(u)

        # Candidate pairs sharing at least one entity
        candidate_pairs = set()
        for u_set in ent_to_users.values():
            if len(u_set) > 1:
                u_list = list(u_set)
                for i in range(len(u_list)):
                    for j in range(i + 1, len(u_list)):
                        pair = (u_list[i], u_list[j]) if u_list[i] < u_list[j] else (u_list[j], u_list[i])
                        candidate_pairs.add(pair)

        records = []
        for u1, u2 in candidate_pairs:
            s1 = active_users[u1]
            s2 = active_users[u2]
            union = s1 | s2
            intersection = s1 & s2
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > 0:
                records.append({"user_a": u1, "user_b": u2, "entity_jaccard": float(jaccard)})

        return pd.DataFrame(records) if records else pd.DataFrame(
            columns=["user_a", "user_b", "entity_jaccard"]
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Core extraction logic."""
        if "text" not in df.columns:
            logger.error("DataFrame missing 'text' column.")
            return pd.DataFrame()

        if "user_id" not in df.columns:
            df = df.copy()
            df["user_id"] = "all_users"

        all_unique_uids = [str(u).strip() for u in df["user_id"].unique()]

        nlp = _get_nlp(self.spacy_model)

        # Per-post entity extraction using spaCy pipe for speed
        texts = df["text"].fillna("").astype(str).tolist()
        user_ids = [str(u).strip() for u in df["user_id"].tolist()]

        post_entity_records = []
        logger.info("NER: processing %d texts with spaCy pipe …", len(texts))

        for doc, uid in zip(
            nlp.pipe(texts, batch_size=self.batch_size, disable=["parser"]),
            user_ids,
        ):
            entities = [
                (ent.text.strip().lower(), ent.label_)
                for ent in doc.ents
                if ent.label_ in self.entity_types
            ]
            post_entity_records.append({"user_id": uid, "entities": entities})

        # Initialize containers for ALL users to guarantee no missing rows
        user_entity_agg: Dict[str, Dict] = {
            uid: {t: Counter() for t in self.entity_types} for uid in all_unique_uids
        }
        user_all_entities: Dict[str, Counter] = {
            uid: Counter() for uid in all_unique_uids
        }

        for record in post_entity_records:
            uid = record["user_id"]
            for ent_text, ent_label in record["entities"]:
                user_entity_agg[uid][ent_label][ent_text] += 1
                user_all_entities[uid][ent_text] += 1

        # Build feature rows
        feature_rows = []
        for uid in all_unique_uids:
            row = {"user_id": uid}

            total_entities = sum(
                sum(c.values()) for c in user_entity_agg[uid].values()
            )
            unique_entities = len(user_all_entities[uid])

            row["ner_total_entities"] = total_entities
            row["ner_unique_entities"] = unique_entities
            row["ner_entity_diversity"] = (
                unique_entities / total_entities if total_entities > 0 else 0.0
            )

            # Per-type counts
            for etype in self.DEFAULT_ENTITY_TYPES:
                count = sum(user_entity_agg[uid][etype].values())
                row[f"ner_{etype.lower()}_count"] = count
                row[f"ner_{etype.lower()}_unique"] = len(user_entity_agg[uid][etype])

            # Watchlist score: fraction of user's entities that are on watchlist
            watchlist_hits = sum(
                cnt
                for ent, cnt in user_all_entities[uid].items()
                if any(w in ent for w in self.watchlist)
            )
            row["ner_watchlist_score"] = (
                watchlist_hits / total_entities if total_entities > 0 else 0.0
            )

            # Concentration: Herfindahl-Hirschman Index on entity types
            type_totals = np.array([
                sum(user_entity_agg[uid][et].values())
                for et in self.DEFAULT_ENTITY_TYPES
            ], dtype=float)
            total_t = type_totals.sum()
            if total_t > 0:
                shares = type_totals / total_t
                row["ner_entity_type_hhi"] = float(np.sum(shares ** 2))
            else:
                row["ner_entity_type_hhi"] = 0.0

            feature_rows.append(row)

        features_df = pd.DataFrame(feature_rows).fillna(0)

        # Append per-user Jaccard coordination scores (symmetric)
        jaccard_df = self.compute_entity_overlap(user_all_entities)
        if not jaccard_df.empty:
            # Symmetrize so both user_a and user_b get scores
            rev_df = jaccard_df.rename(columns={"user_a": "user_b", "user_b": "user_a"})
            both_sides = pd.concat([jaccard_df, rev_df], ignore_index=True)
            avg_jaccard = (
                both_sides.groupby("user_a")["entity_jaccard"]
                .mean()
                .reset_index()
                .rename(columns={"user_a": "user_id", "entity_jaccard": "ner_avg_entity_jaccard"})
            )
            features_df = features_df.merge(avg_jaccard, on="user_id", how="left")
            features_df["ner_avg_entity_jaccard"] = features_df["ner_avg_entity_jaccard"].fillna(0.0)
        else:
            features_df["ner_avg_entity_jaccard"] = 0.0

        logger.info("NER features extracted for %d users", len(features_df))
        return features_df


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_data = pd.DataFrame(
        {
            "user_id": ["bot_1", "bot_1", "bot_2", "bot_2", "human_1", "human_1"],
            "text": [
                "George Soros and the CIA are funding Antifa to destabilise the US government.",
                "Breaking: The FBI confirmed that Soros donated to the DNC. RT if true!",
                "George Soros & CIA behind the chaos. Wake up America!",
                "FBI and Soros are controlling the media. Share this everywhere!",
                "Just watched a great documentary about NASA's Mars mission.",
                "Apple's new iPhone looks amazing. Tim Cook gave a fantastic keynote.",
            ],
        }
    )

    extractor = NERFeatureExtractor()
    features = extractor.extract_from_dataframe(sample_data)
    print("\n=== NER Features ===")
    print(features.to_string(index=False))
