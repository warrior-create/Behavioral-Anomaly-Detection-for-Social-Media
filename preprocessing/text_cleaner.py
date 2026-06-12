"""
Text Cleaning and Preprocessing Module
Handles text normalization, language detection, and data quality
"""

import pandas as pd
import re
import logging
from typing import Optional
import langdetect
from langdetect import DetectorFactory
import unicodedata

# Set seed for reproducible language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)


class TextCleaner:
    """Cleans and preprocesses social media text data"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize text cleaner
        
        Args:
            config: Configuration dictionary for preprocessing options
        """
        self.config = config or {}
        self.stats = {
            'total_processed': 0,
            'filtered_language': 0,
            'filtered_length': 0,
            'cleaned': 0
        }
        logger.info("Text Cleaner initialized")
    
    def process_file(self, file_path: str) -> pd.DataFrame:
        """
        Process entire CSV/JSON file
        
        Args:
            file_path: Path to input file
        
        Returns:
            Cleaned DataFrame
        """
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, lines=True)
            else:
                logger.error(f"Unsupported file format: {file_path}")
                return pd.DataFrame()
            
            logger.info(f"Processing {len(df)} records from {file_path}")
            self.stats['total_processed'] = len(df)
            
            # Clean text column
            if 'text' in df.columns:
                df['text_original'] = df['text'].copy()
                df['text'] = df['text'].apply(self.clean_text)
                
                # Language detection
                if self.config.get('detect_language', True):
                    df['language'] = df['text'].apply(self.detect_language)
                    
                    # Filter by language
                    target_lang = self.config.get('language_filter', 'en')
                    if target_lang:
                        before_count = len(df)
                        df = df[df['language'] == target_lang]
                        self.stats['filtered_language'] = before_count - len(df)
                        logger.info(f"Filtered {self.stats['filtered_language']} non-{target_lang} texts")
                
                # Filter by minimum length
                min_length = self.config.get('min_text_length', 10)
                before_count = len(df)
                df = df[df['text'].str.len() >= min_length]
                self.stats['filtered_length'] = before_count - len(df)
                logger.info(f"Filtered {self.stats['filtered_length']} short texts")
                
                self.stats['cleaned'] = len(df)
                logger.info(f"Cleaning complete: {len(df)} records remaining")
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text with configurable options
        
        Args:
            text: Raw text string
        
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove URLs
        if self.config.get('remove_urls', True):
            text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove mentions (Twitter @mentions)
        if self.config.get('remove_mentions', True):
            text = re.sub(r'@\w+', '', text)
        
        # Handle hashtags
        if self.config.get('remove_hashtags', False):
            text = re.sub(r'#\w+', '', text)
        else:
            # Keep hashtags but remove # symbol
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle emojis
        if self.config.get('remove_emojis', False):
            text = self.remove_emojis(text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Remove extra punctuation (but keep some for sentiment)
        if self.config.get('remove_punctuation', False):
            text = re.sub(r'[^\w\s]', '', text)
        else:
            # Just normalize repeated punctuation
            text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        # Remove extra whitespace again
        text = ' '.join(text.split())
        
        return text.strip()
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emoji characters from text
        
        Args:
            text: Text containing emojis
        
        Returns:
            Text without emojis
        """
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
        
        Returns:
            ISO 639-1 language code (e.g., 'en', 'es')
        """
        try:
            if not text or len(text) < 10:
                return 'unknown'
            lang = langdetect.detect(text)
            return lang
        except Exception:
            return 'unknown'
    
    def extract_hashtags(self, text: str) -> list:
        """
        Extract hashtags from text
        
        Args:
            text: Text containing hashtags
        
        Returns:
            List of hashtags (without #)
        """
        hashtags = re.findall(r'#(\w+)', text)
        return hashtags
    
    def extract_mentions(self, text: str) -> list:
        """
        Extract mentions from text
        
        Args:
            text: Text containing mentions
        
        Returns:
            List of usernames (without @)
        """
        mentions = re.findall(r'@(\w+)', text)
        return mentions
    
    def extract_urls(self, text: str) -> list:
        """
        Extract URLs from text
        
        Args:
            text: Text containing URLs
        
        Returns:
            List of URLs
        """
        urls = re.findall(r'http\S+|www\.\S+', text)
        return urls
    
    def get_cleaning_stats(self) -> dict:
        """
        Get statistics about cleaning process
        
        Returns:
            Dictionary with cleaning statistics
        """
        return self.stats
    
    def batch_clean(self, texts: list) -> list:
        """
        Clean multiple texts efficiently
        
        Args:
            texts: List of text strings
        
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize various types of whitespace
        
        Args:
            text: Text with irregular whitespace
        
        Returns:
            Text with normalized whitespace
        """
        # Replace various whitespace chars with standard space
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    def remove_html_tags(self, text: str) -> str:
        """
        Remove HTML tags from text
        
        Args:
            text: Text containing HTML
        
        Returns:
            Text without HTML tags
        """
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def expand_contractions(self, text: str) -> str:
        """
        Expand common English contractions
        
        Args:
            text: Text with contractions
        
        Returns:
            Text with expanded contractions
        """
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'ve": " have",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Check out this link: https://example.com #AI #MachineLearning @user123",
        "This is AWESOME!!! 😀😎 Can't wait!!!",
        "RT @someone: Important announcement",
        "   Extra    whitespace   everywhere   "
    ]
    
    config = {
        'remove_urls': True,
        'remove_mentions': True,
        'remove_hashtags': False,
        'remove_emojis': False,
        'lowercase': True
    }
    
    cleaner = TextCleaner(config)
    
    print("Text Cleaning Examples:")
    print("-" * 80)
    for text in test_texts:
        cleaned = cleaner.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Language: {cleaner.detect_language(text)}")
        print("-" * 80)