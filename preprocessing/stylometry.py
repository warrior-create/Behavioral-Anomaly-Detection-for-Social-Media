"""
Stylometric Feature Extraction
Analyzes writing style characteristics for bot and coordination detection
"""

import pandas as pd
import numpy as np
import logging
import re
from collections import Counter
from typing import Dict, List
from textblob import TextBlob

logger = logging.getLogger(__name__)


class StylometricAnalyzer:
    """Extracts linguistic and stylistic features from text"""
    
    def __init__(self):
        """Initialize stylometric analyzer"""
        logger.info("Stylometric Analyzer initialized")
    
    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Extract stylometric features from CSV file
        
        Args:
            file_path: Path to cleaned CSV file
        
        Returns:
            DataFrame with stylometric features
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Extracting stylometric features from {len(df)} texts")
            
            # Apply feature extraction to each text
            features_list = []
            for idx, row in df.iterrows():
                text = row.get('text', '')
                features = self.analyze_text(text)
                features['user_id'] = row.get('user_id', '')
                features['post_id'] = row.get('post_id', '')
                features_list.append(features)
            
            features_df = pd.DataFrame(features_list)
            
            # Aggregate by user
            user_features = self._aggregate_user_features(features_df)
            
            logger.info(f"Extracted features for {len(user_features)} users")
            return user_features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze single text for stylometric features
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary of features
        """
        if not isinstance(text, str) or not text:
            return self._empty_features()
        
        features = {}
        
        # Basic length metrics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        
        # Sentence metrics
        sentences = self._split_sentences(text)
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = (
            features['word_count'] / features['sentence_count']
            if features['sentence_count'] > 0 else 0
        )
        
        # Word length metrics
        words = text.split()
        if words:
            word_lengths = [len(word) for word in words]
            features['avg_word_length'] = np.mean(word_lengths)
            features['word_length_std'] = np.std(word_lengths)
        else:
            features['avg_word_length'] = 0
            features['word_length_std'] = 0
        
        # Vocabulary richness (Type-Token Ratio)
        features['vocabulary_richness'] = self._calculate_ttr(text)
        
        # Hapax legomena ratio (words appearing only once)
        features['hapax_ratio'] = self._calculate_hapax_ratio(text)
        
        # Punctuation metrics
        features['punctuation_ratio'] = self._calculate_punctuation_ratio(text)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        
        # Capitalization metrics
        features['uppercase_ratio'] = self._calculate_uppercase_ratio(text)
        features['capitalized_words'] = sum(1 for word in words if word and word[0].isupper())
        
        # Special character counts
        features['hashtag_count'] = text.count('#')
        features['mention_count'] = text.count('@')
        features['url_count'] = len(re.findall(r'http\S+|www\.\S+', text))
        
        # Emoji detection and counting
        features['emoji_count'] = self._count_emojis(text)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(text)
        features['sentiment_polarity'] = sentiment['polarity']
        features['sentiment_subjectivity'] = sentiment['subjectivity']
        
        # Lexical diversity
        features['lexical_diversity'] = self._calculate_lexical_diversity(text)
        
        # Function word ratio (common words like 'the', 'a', 'is')
        features['function_word_ratio'] = self._calculate_function_word_ratio(text)
        
        # Repetition metrics
        features['word_repetition_score'] = self._calculate_repetition_score(text)
        
        return features
    
    def _empty_features(self) -> Dict:
        """Return dictionary of zero-valued features"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_sentence_length': 0, 'avg_word_length': 0, 'word_length_std': 0,
            'vocabulary_richness': 0, 'hapax_ratio': 0, 'punctuation_ratio': 0,
            'exclamation_count': 0, 'question_count': 0, 'ellipsis_count': 0,
            'uppercase_ratio': 0, 'capitalized_words': 0, 'hashtag_count': 0,
            'mention_count': 0, 'url_count': 0, 'emoji_count': 0,
            'sentiment_polarity': 0, 'sentiment_subjectivity': 0,
            'lexical_diversity': 0, 'function_word_ratio': 0,
            'word_repetition_score': 0
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_ttr(self, text: str) -> float:
        """Calculate Type-Token Ratio (vocabulary richness)"""
        words = text.lower().split()
        if not words:
            return 0
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _calculate_hapax_ratio(self, text: str) -> float:
        """Calculate ratio of words appearing only once"""
        words = text.lower().split()
        if not words:
            return 0
        word_freq = Counter(words)
        hapax = sum(1 for freq in word_freq.values() if freq == 1)
        return hapax / len(words)
    
    def _calculate_punctuation_ratio(self, text: str) -> float:
        """Calculate ratio of punctuation characters"""
        if not text:
            return 0
        punct = sum(1 for char in text if char in '.,!?;:')
        return punct / len(text)
    
    def _calculate_uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase letters"""
        if not text:
            return 0
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return 0
        uppercase = sum(1 for char in letters if char.isupper())
        return uppercase / len(letters)
    
    def _count_emojis(self, text: str) -> int:
        """Count emoji characters in text"""
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        return len(emoji_pattern.findall(text))
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception:
            return {'polarity': 0, 'subjectivity': 0}
    
    def _calculate_lexical_diversity(self, text: str) -> float:
        """Calculate lexical diversity (Yule's K measure)"""
        words = text.lower().split()
        if len(words) < 2:
            return 0
        
        word_freq = Counter(words)
        M1 = len(words)
        M2 = sum([freq ** 2 for freq in word_freq.values()])
        
        try:
            return (M1 * M1) / (M2 - M1)
        except ZeroDivisionError:
            return 0
    
    def _calculate_function_word_ratio(self, text: str) -> float:
        """Calculate ratio of function words"""
        function_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'of', 'in',
            'to', 'for', 'with', 'on', 'at', 'by', 'from', 'as'
        }
        
        words = text.lower().split()
        if not words:
            return 0
        
        function_count = sum(1 for word in words if word in function_words)
        return function_count / len(words)
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate word repetition score"""
        words = text.lower().split()
        if len(words) < 2:
            return 0
        
        # Count consecutive repeated words
        repetitions = sum(
            1 for i in range(len(words) - 1)
            if words[i] == words[i + 1]
        )
        return repetitions / len(words)
    
    def _aggregate_user_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate post-level features to user-level
        
        Args:
            features_df: DataFrame with post-level features
        
        Returns:
            DataFrame with user-level aggregated features
        """
        if 'user_id' not in features_df.columns:
            return features_df
        
        # Aggregate numeric features by user
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'user_id']
        
        aggregated = features_df.groupby('user_id')[numeric_cols].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            '_'.join(col).strip('_') if col[1] else col[0]
            for col in aggregated.columns.values
        ]
        
        return aggregated


# Example usage
if __name__ == "__main__":
    # Test text
    test_texts = [
        "This is a test! Can you believe it? #amazing @user",
        "wow this is GREAT!!! 😀 So happy!!!",
        "Check this out: https://example.com Very interesting."
    ]
    
    analyzer = StylometricAnalyzer()
    
    print("Stylometric Analysis Examples:")
    print("=" * 80)
    for text in test_texts:
        features = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Features: {features}")