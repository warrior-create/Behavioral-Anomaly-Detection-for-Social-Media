# ==============================================================================
# ingestion/__init__.py
# ==============================================================================
"""
Data Ingestion Module
Handles data collection from social media platforms (Twitter, Reddit)
"""

from .twitter_collector import TwitterCollector

__all__ = ['TwitterCollector']

