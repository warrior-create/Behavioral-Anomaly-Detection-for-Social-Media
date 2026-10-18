"""
Text Normalizer — NLP Preprocessing Extension
==============================================
Extends the basic TextCleaner with full NLP preprocessing:
  - Tokenization (NLTK word_tokenize)
  - Stop-word removal
  - Lemmatization (WordNetLemmatizer)
  - Stemming (PorterStemmer, optional alternative to lemmatization)
  - POS-tag filtering (keep only nouns, verbs, adjectives)
  - Slang / abbreviation expansion
  - Numeric and repeated-character normalization

Why this matters:
  Better normalisation directly improves TF-IDF, PMI, LDA, and perplexity
  features because the vocabulary is cleaner (e.g. "running" and "runs"
  both reduce to "run") and stop words don't dominate TF-IDF weights.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy NLTK imports with clear instructions on first failure
_NLTK_READY = False
_word_tokenize = None
_stopwords_set = None
_lemmatizer = None
_stemmer = None


def _ensure_nltk():
    """Download and cache required NLTK resources on first use."""
    global _NLTK_READY, _word_tokenize, _stopwords_set, _lemmatizer, _stemmer

    if _NLTK_READY:
        return

    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk import pos_tag

        # Download required data (idempotent)
        for resource in ["punkt", "punkt_tab", "stopwords", "wordnet",
                         "omw-1.4", "averaged_perceptron_tagger"]:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass

        _word_tokenize = word_tokenize
        _stopwords_set = set(stopwords.words("english"))
        _lemmatizer    = WordNetLemmatizer()
        _stemmer       = PorterStemmer()
        _NLTK_READY    = True
        logger.info("NLTK resources loaded.")

    except ImportError:
        logger.error("NLTK not installed. Run: pip install nltk")
        raise


# Common social-media slang expansions
_SLANG_MAP = {
    r"\bbtw\b": "by the way",
    r"\bimo\b": "in my opinion",
    r"\bidk\b": "i do not know",
    r"\blol\b": "laughing",
    r"\bomg\b": "oh my god",
    r"\bafaik\b": "as far as i know",
    r"\bsmh\b": "shaking my head",
    r"\bfwiw\b": "for what it is worth",
    r"\btbh\b": "to be honest",
    r"\bngl\b": "not gonna lie",
    r"\birl\b": "in real life",
    r"\brt\b": "retweet",
    r"\bdm\b": "direct message",
    r"\bfomo\b": "fear of missing out",
}

# POS tags to keep when filtering (None = keep all)
_KEEP_POS = {"NN", "NNS", "NNP", "NNPS",   # Nouns
             "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # Verbs
             "JJ", "JJR", "JJS",            # Adjectives
             "RB", "RBR", "RBS"}            # Adverbs


def _get_wordnet_pos(tag: str) -> str:
    """Map NLTK POS tag to WordNet POS tag for lemmatization."""
    from nltk.corpus import wordnet
    tag = tag[0].upper()
    tag_map = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }
    return tag_map.get(tag, wordnet.NOUN)


class TextNormalizer:
    """
    Full NLP text normalisation pipeline.

    Steps (configurable):
      1. Lowercase
      2. Slang expansion
      3. Numeric normalisation (e.g. "100000" → "<NUM>")
      4. Repeated-character collapse ("loooove" → "love")
      5. Tokenization (NLTK word_tokenize)
      6. Stop-word removal
      7. POS-tag filtering (optional)
      8. Lemmatization (default) or Stemming (optional)
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialise text normalizer.

        Args:
            config: Optional dict with keys:
                - remove_stopwords  (bool, default True)
                - lemmatize         (bool, default True)
                - stem              (bool, default False)
                - filter_pos        (bool, default False) — keep only content words
                - expand_slang      (bool, default True)
                - normalise_numbers (bool, default True)
                - collapse_repeated (bool, default True)
                - min_token_length  (int, default 2)
        """
        self.config = config or {}
        self.remove_stopwords  = self.config.get("remove_stopwords",  True)
        self.lemmatize         = self.config.get("lemmatize",         True)
        self.stem              = self.config.get("stem",              False)
        self.filter_pos        = self.config.get("filter_pos",        False)
        self.expand_slang      = self.config.get("expand_slang",      True)
        self.normalise_numbers = self.config.get("normalise_numbers", True)
        self.collapse_repeated = self.config.get("collapse_repeated", True)
        self.min_token_length  = self.config.get("min_token_length",  2)

        # Lemmatizer wins over stemmer if both are True
        if self.lemmatize and self.stem:
            logger.warning("Both lemmatize and stem set — lemmatize takes priority.")
            self.stem = False

        _ensure_nltk()
        logger.info(
            "TextNormalizer | lemmatize=%s | stem=%s | rm_stop=%s | filter_pos=%s",
            self.lemmatize, self.stem, self.remove_stopwords, self.filter_pos,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def normalize(self, text: str) -> str:
        """
        Run the full normalisation pipeline on a single string.

        Args:
            text: Raw or lightly-cleaned text.

        Returns:
            Normalised, space-joined token string ready for NLP models.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()

        if self.expand_slang:
            text = self._expand_slang(text)

        if self.normalise_numbers:
            text = re.sub(r"\b\d+\b", "<NUM>", text)

        if self.collapse_repeated:
            # "loooove" → "loove" (collapse runs > 2)
            text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        tokens = _word_tokenize(text)

        if self.filter_pos:
            tokens = self._pos_filter(tokens)

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in _stopwords_set]

        # Length filter
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        if self.lemmatize:
            tokens = self._lemmatize(tokens)
        elif self.stem:
            tokens = [_stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """Normalize a list of texts."""
        return [self.normalize(t) for t in texts]

    def tokenize(self, text: str) -> List[str]:
        """Return normalised token list (not joined string)."""
        return self.normalize(text).split()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _expand_slang(self, text: str) -> str:
        for pattern, replacement in _SLANG_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _pos_filter(self, tokens: List[str]) -> List[str]:
        import nltk
        tagged = nltk.pos_tag(tokens)
        return [word for word, tag in tagged if tag in _KEEP_POS]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        import nltk
        tagged = nltk.pos_tag(tokens)
        return [
            _lemmatizer.lemmatize(word, pos=_get_wordnet_pos(tag))
            for word, tag in tagged
        ]


# ---------------------------------------------------------------------------
# CLI / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "OMG lol this is AMAZING!!! 😂😂 Win FREE Bitcoin giveaway NOW btw",
        "The government is totally covering this UP smh wake UP people",
        "I was running and ran into my friend who runs a small business",
        "BREEEEAKING NEWS: Prices are 100000 times higher than last year!!",
    ]

    normalizer = TextNormalizer(config={
        "lemmatize": True,
        "remove_stopwords": True,
        "expand_slang": True,
        "filter_pos": False,
    })

    print("=== Text Normalizer Demo ===\n")
    for s in samples:
        print(f"  Original : {s}")
        print(f"  Normalized: {normalizer.normalize(s)}\n")
