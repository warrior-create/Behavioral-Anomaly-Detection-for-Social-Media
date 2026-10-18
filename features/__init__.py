# ==============================================================================
# features/__init__.py
# ==============================================================================
"""
Feature Extraction Module
Exports all feature extractors:
  - BERT semantic embeddings
  - Temporal behaviour features
  - TF-IDF content similarity (NEW)
  - Named Entity Recognition (NEW)
  - LDA Topic Modelling (NEW)
  - N-gram Perplexity / AI-text detection (NEW)
  - PMI Keyword Co-occurrence (NEW)
"""

from .bert_embeddings import BERTEmbedder
from .temporal_features import TemporalFeatureExtractor
from .tfidf_features import TFIDFAnalyzer
from .ner_features import NERFeatureExtractor
from .topic_modeling import TopicModeler
from .ngram_perplexity import PerplexityFeatureExtractor
from .pmi_cooccurrence import PMICooccurrenceAnalyzer

__all__ = [
    'BERTEmbedder',
    'TemporalFeatureExtractor',
    'TFIDFAnalyzer',
    'NERFeatureExtractor',
    'TopicModeler',
    'PerplexityFeatureExtractor',
    'PMICooccurrenceAnalyzer',
]