"""
Temporal Feature Extraction
Analyzes posting patterns and temporal coordination
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from scipy import stats

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """Extracts temporal behavioral features"""
    
    def __init__(self):
        """Initialize temporal feature extractor"""
        logger.info("Temporal Feature Extractor initialized")
    
    def extract_features(self, file_path: str) -> pd.DataFrame:
        """
        Extract temporal features from dataset
        
        Args:
            file_path: Path to CSV with timestamp column
        
        Returns:
            DataFrame with temporal features per user
        """
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Extracting temporal features for {len(df)} posts")
            
            # Group by user and extract features
            # We reset index to ensure user_id is a column
            user_features = df.groupby('user_id').apply(
                lambda x: pd.Series(self._compute_user_temporal_features(x))
            ).reset_index()
            
            logger.info(f"Extracted temporal features for {len(user_features)} users")
            return user_features
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return pd.DataFrame()
    
    def _compute_user_temporal_features(self, user_df: pd.DataFrame) -> Dict:
        """
        Compute temporal features for a single user
        
        Args:
            user_df: DataFrame with user's posts
        
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Sort by timestamp
        user_df = user_df.sort_values('timestamp')
        timestamps = user_df['timestamp'].values
        
        # Basic posting statistics
        features['total_posts'] = len(user_df)
        
        # --- FIX START: Correctly calculate duration in days ---
        if len(timestamps) > 0:
            # Divide the time difference by 1 Day to get a float value
            duration_days = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, 'D')
            features['posting_days'] = int(duration_days) + 1
        else:
            features['posting_days'] = 0
        # --- FIX END ---

        features['posts_per_day'] = features['total_posts'] / max(features['posting_days'], 1)
        
        # Inter-post time intervals
        if len(timestamps) > 1:
            intervals = np.diff(timestamps) / np.timedelta64(1, 's')  # Convert to seconds
            features['avg_inter_post_time'] = np.mean(intervals)
            features['std_inter_post_time'] = np.std(intervals)
            features['min_inter_post_time'] = np.min(intervals)
            features['max_inter_post_time'] = np.max(intervals)
            
            # Very short intervals might indicate bot behavior
            features['burst_ratio'] = np.sum(intervals < 60) / len(intervals)  # Posts within 1 minute
        else:
            features['avg_inter_post_time'] = 0
            features['std_inter_post_time'] = 0
            features['min_inter_post_time'] = 0
            features['max_inter_post_time'] = 0
            features['burst_ratio'] = 0
        
        # Posting time distribution (entropy)
        features['posting_time_entropy'] = self._calculate_time_entropy(timestamps)
        
        # Hour of day distribution
        hour_dist = self._get_hour_distribution(timestamps)
        features['night_posting_ratio'] = hour_dist.get('night_ratio', 0)
        features['business_hours_ratio'] = hour_dist.get('business_ratio', 0)
        features['peak_hour'] = hour_dist.get('peak_hour', 0)
        
        # Day of week distribution
        day_dist = self._get_day_distribution(timestamps)
        features['weekend_posting_ratio'] = day_dist.get('weekend_ratio', 0)
        features['weekday_posting_ratio'] = day_dist.get('weekday_ratio', 0)
        
        # Regularity measures
        features['posting_regularity'] = self._calculate_posting_regularity(timestamps)
        
        # Activity burst detection
        burst_info = self._detect_activity_bursts(timestamps)
        features['num_bursts'] = burst_info['num_bursts']
        features['avg_burst_size'] = burst_info['avg_burst_size']
        
        return features
    
    def _calculate_time_entropy(self, timestamps: np.ndarray) -> float:
        """
        Calculate entropy of posting times (higher = more random, lower = more regular)
        """
        if len(timestamps) == 0:
            return 0
        
        # Convert to pandas datetime to access .hour accessor safely
        ts_series = pd.to_datetime(timestamps)
        hours = ts_series.hour
        hour_counts = np.bincount(hours, minlength=24)
        
        # Calculate probability distribution
        probs = hour_counts / len(timestamps)
        probs = probs[probs > 0]  # Remove zero probabilities
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def _get_hour_distribution(self, timestamps: np.ndarray) -> Dict:
        """Analyze hour-of-day posting distribution"""
        if len(timestamps) == 0:
            return {'night_ratio': 0, 'business_ratio': 0, 'peak_hour': 0}
        
        ts_series = pd.to_datetime(timestamps)
        hours = ts_series.hour
        hour_counts = np.bincount(hours, minlength=24)
        
        # Night hours (midnight to 5 AM)
        night_posts = sum(hour_counts[0:6])
        night_ratio = night_posts / len(timestamps)
        
        # Business hours (9 AM to 5 PM)
        business_posts = sum(hour_counts[9:17])
        business_ratio = business_posts / len(timestamps)
        
        # Peak hour
        peak_hour = np.argmax(hour_counts)
        
        return {
            'night_ratio': night_ratio,
            'business_ratio': business_ratio,
            'peak_hour': peak_hour
        }
    
    def _get_day_distribution(self, timestamps: np.ndarray) -> Dict:
        """Analyze day-of-week posting distribution"""
        if len(timestamps) == 0:
            return {'weekend_ratio': 0, 'weekday_ratio': 0}
        
        ts_series = pd.to_datetime(timestamps)
        days = ts_series.dayofweek
        
        # Weekend (Saturday=5, Sunday=6)
        weekend_posts = sum(1 for d in days if d >= 5)
        weekend_ratio = weekend_posts / len(timestamps)
        
        weekday_ratio = 1 - weekend_ratio
        
        return {
            'weekend_ratio': weekend_ratio,
            'weekday_ratio': weekday_ratio
        }
    
    def _calculate_posting_regularity(self, timestamps: np.ndarray) -> float:
        """Calculate posting regularity score"""
        if len(timestamps) < 2:
            return 0
        
        intervals = np.diff(timestamps) / np.timedelta64(1, 's')
        
        if np.mean(intervals) == 0:
            return 0
        
        # Coefficient of variation
        cv = np.std(intervals) / np.mean(intervals)
        
        # Convert to regularity score (inverse and normalize)
        regularity = 1 / (1 + cv)
        return regularity
    
    def _detect_activity_bursts(
        self,
        timestamps: np.ndarray,
        time_window: int = 300
    ) -> Dict:
        """Detect bursts of activity (many posts in short time)"""
        if len(timestamps) < 2:
            return {'num_bursts': 0, 'avg_burst_size': 0}
        
        timestamps_sorted = np.sort(timestamps)
        bursts = []
        current_burst = [timestamps_sorted[0]]
        
        for i in range(1, len(timestamps_sorted)):
            time_diff = (timestamps_sorted[i] - current_burst[-1]) / np.timedelta64(1, 's')
            
            if time_diff <= time_window:
                current_burst.append(timestamps_sorted[i])
            else:
                if len(current_burst) >= 3:  # Minimum 3 posts for a burst
                    bursts.append(len(current_burst))
                current_burst = [timestamps_sorted[i]]
        
        # Check last burst
        if len(current_burst) >= 3:
            bursts.append(len(current_burst))
        
        return {
            'num_bursts': len(bursts),
            'avg_burst_size': np.mean(bursts) if bursts else 0
        }
    
    def detect_synchronized_posting(
        self,
        df: pd.DataFrame,
        time_window: int = 60,
        min_users: int = 3
    ) -> pd.DataFrame:
        """Detect groups of users posting within same time windows"""
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            synchronized_events = []
            
            for i in range(len(df)):
                window_start = df.iloc[i]['timestamp']
                window_end = window_start + timedelta(seconds=time_window)
                
                # Find posts within window
                mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
                window_df = df[mask]
                
                unique_users = window_df['user_id'].nunique()
                
                if unique_users >= min_users:
                    synchronized_events.append({
                        'window_start': window_start,
                        'window_end': window_end,
                        'num_users': unique_users,
                        'num_posts': len(window_df),
                        'users': window_df['user_id'].unique().tolist()
                    })
            
            result_df = pd.DataFrame(synchronized_events)
            logger.info(f"Detected {len(result_df)} synchronized posting events")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error detecting synchronized posting: {e}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Create test data
    test_data = pd.DataFrame({
        'user_id': ['user1'] * 10 + ['user2'] * 10,
        'timestamp': pd.date_range('2024-01-01', periods=20, freq='1H'),
        'text': ['test'] * 20
    })
    
    extractor = TemporalFeatureExtractor()
    
    # Extract features
    features = extractor.extract_features('test_data.csv')
    print("Temporal Features:")
    print(features.head())