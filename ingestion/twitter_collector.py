"""
Twitter Data Collector - Standalone Version
Collects data in exact format needed for CIB detection
"""

import tweepy
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class TwitterCollector:
    """Collect tweets in standardized format for CIB detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        self.tweet_count = 0
        
        if config.get('enabled', False):
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Twitter API v2"""
        try:
            self.client = tweepy.Client(
                bearer_token=self.config.get('bearer_token'),
                consumer_key=self.config.get('api_key'),
                consumer_secret=self.config.get('api_secret'),
                access_token=self.config.get('access_token'),
                access_token_secret=self.config.get('access_token_secret'),
                wait_on_rate_limit=True
            )
            
            # Test authentication
            me = self.client.get_me()
            logger.info(f"✓ Twitter authentication successful")
            logger.info(f"  Authenticated as: @{me.data.username}")
            
        except tweepy.Unauthorized:
            logger.error("✗ Authentication failed - Check your API credentials")
            raise
        except tweepy.Forbidden:
            logger.error("✗ Access forbidden - Your app may not have proper permissions")
            raise
        except Exception as e:
            logger.error(f"✗ Twitter authentication failed: {e}")
            raise
    
    def collect_data(self) -> pd.DataFrame:
        """
        Main method to collect tweets based on configuration
        
        Returns:
            DataFrame with columns: user_id, post_id, text, timestamp, reply_to, mentions, hashtags, platform
        """
        logger.info("Starting Twitter data collection...")
        
        all_tweets = []
        keywords = self.config.get('keywords', [])
        max_results = self.config.get('max_results', 1000)
        
        # Build search query
        query = self._build_search_query(keywords)
        logger.info(f"Search query: {query}")
        
        try:
            # Collect tweets
            tweets = self._search_tweets(query, max_results)
            all_tweets.extend(tweets)
            
            logger.info(f"✓ Collected {len(all_tweets)} tweets")
            
        except Exception as e:
            logger.error(f"Error during collection: {e}")
        
        # Convert to DataFrame
        if all_tweets:
            df = self._convert_to_dataframe(all_tweets)
            return df
        else:
            logger.warning("No tweets collected")
            return pd.DataFrame(columns=[
                'user_id', 'post_id', 'text', 'timestamp', 
                'reply_to', 'mentions', 'hashtags', 'platform'
            ])
    
    def collect_real_time(self, duration_minutes: int = 60) -> pd.DataFrame:
        """
        Collect tweets in real-time for specified duration
        
        Args:
            duration_minutes: How long to collect (in minutes)
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info("=" * 80)
        logger.info("REAL-TIME TWITTER COLLECTION")
        logger.info("=" * 80)
        logger.info(f"Duration: {duration_minutes} minutes")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_tweets = []
        keywords = self.config.get('keywords', [])
        query = self._build_search_query(keywords)
        
        end_time = datetime.now().timestamp() + (duration_minutes * 60)
        last_tweet_id = None
        collection_round = 0
        
        while datetime.now().timestamp() < end_time:
            collection_round += 1
            logger.info(f"\n--- Collection Round {collection_round} ---")
            
            try:
                # Search for recent tweets
                tweets = self.client.search_recent_tweets(
                    query=query,
                    max_results=100,
                    since_id=last_tweet_id,  # Only get tweets newer than last collection
                    tweet_fields=['id', 'text', 'author_id', 'created_at', 
                                  'public_metrics', 'referenced_tweets', 
                                  'entities', 'lang'],
                    user_fields=['id', 'username'],
                    expansions=['author_id', 'referenced_tweets.id']
                )
                
                if tweets.data:
                    # Process tweets
                    for tweet in tweets.data:
                        tweet_data = self._extract_tweet_data(tweet)
                        all_tweets.append(tweet_data)
                    
                    # Update last seen tweet ID
                    last_tweet_id = tweets.data[0].id
                    
                    logger.info(f"  ✓ Collected {len(tweets.data)} new tweets")
                    logger.info(f"  Total so far: {len(all_tweets)}")
                else:
                    logger.info(f"  No new tweets in this round")
                
                # Calculate remaining time
                remaining = int((end_time - datetime.now().timestamp()) / 60)
                logger.info(f"  Time remaining: {remaining} minutes")
                
                # Wait before next request (avoid rate limits)
                time.sleep(20)  # 20 seconds between requests
                
            except tweepy.TooManyRequests:
                logger.warning("⚠ Rate limit reached, waiting 15 minutes...")
                time.sleep(900)  # 15 minutes
                
            except tweepy.TwitterServerError:
                logger.warning("⚠ Twitter server error, waiting 1 minute...")
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"✗ Error during collection: {e}")
                time.sleep(60)
        
        logger.info("=" * 80)
        logger.info(f"✓ COLLECTION COMPLETE")
        logger.info(f"Total tweets collected: {len(all_tweets)}")
        logger.info("=" * 80)
        
        # Convert to DataFrame
        if all_tweets:
            return self._convert_to_dataframe(all_tweets)
        else:
            return pd.DataFrame(columns=[
                'user_id', 'post_id', 'text', 'timestamp',
                'reply_to', 'mentions', 'hashtags', 'platform'
            ])
    
    def _build_search_query(self, keywords: List[str]) -> str:
        """Build Twitter search query from keywords"""
        # Combine keywords with OR
        keyword_query = ' OR '.join([f'"{kw}"' for kw in keywords])
        
        # Add filters
        filters = [
            '-is:retweet',  # Exclude retweets
            'lang:en',      # English only
            '-is:reply'     # Exclude replies (we'll get original tweets)
        ]
        
        query = f"({keyword_query}) {' '.join(filters)}"
        
        return query
    
    def _search_tweets(self, query: str, max_results: int) -> List[Dict]:
        """Search tweets using Twitter API v2"""
        tweets = []
        
        try:
            tweet_fields = [
                'id', 'text', 'author_id', 'created_at',
                'public_metrics', 'referenced_tweets', 
                'entities', 'lang', 'conversation_id'
            ]
            
            user_fields = ['id', 'username', 'name']
            expansions = ['author_id', 'referenced_tweets.id']
            
            # Use pagination to get all results
            for response in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                max_results=min(100, max_results),
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=expansions
            ).flatten(limit=max_results):
                
                tweet_data = self._extract_tweet_data(response)
                tweets.append(tweet_data)
                
                if len(tweets) % 100 == 0:
                    logger.info(f"  Collected {len(tweets)} tweets so far...")
            
        except tweepy.TweepyException as e:
            logger.error(f"Error searching tweets: {e}")
        
        return tweets
    
    def _extract_tweet_data(self, tweet) -> Dict:
        """Extract data from tweet object in standardized format"""
        
        # Extract mentions
        mentions = []
        if hasattr(tweet, 'entities') and tweet.entities:
            if 'mentions' in tweet.entities:
                mentions = [m['username'] for m in tweet.entities['mentions']]
        
        # Extract hashtags
        hashtags = []
        if hasattr(tweet, 'entities') and tweet.entities:
            if 'hashtags' in tweet.entities:
                hashtags = [h['tag'] for h in tweet.entities['hashtags']]
        
        # Check if this is a reply
        reply_to = None
        if hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets:
            for ref in tweet.referenced_tweets:
                if ref.type == 'replied_to':
                    reply_to = str(ref.id)
                    break
        
        return {
            'user_id': str(tweet.author_id),
            'post_id': str(tweet.id),
            'text': tweet.text.replace('\n', ' ').replace('\r', ' '),  # Clean text
            'timestamp': tweet.created_at.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'reply_to': reply_to if reply_to else '',
            'mentions': ','.join(mentions) if mentions else '',
            'hashtags': ','.join(['#' + h for h in hashtags]) if hashtags else '',
            'platform': 'twitter'
        }
    
    def _convert_to_dataframe(self, tweets: List[Dict]) -> pd.DataFrame:
        """Convert tweet list to DataFrame with exact required columns"""
        
        df = pd.DataFrame(tweets)
        
        # Ensure all required columns exist
        required_columns = [
            'user_id', 'post_id', 'text', 'timestamp',
            'reply_to', 'mentions', 'hashtags', 'platform'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns to match expected format
        df = df[required_columns]
        
        # Clean data
        df = df.fillna('')
        df['text'] = df['text'].astype(str)
        df['user_id'] = df['user_id'].astype(str)
        df['post_id'] = df['post_id'].astype(str)
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save collected data to CSV"""
        try:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"✓ Data saved to {output_path}")
            logger.info(f"  Total records: {len(df)}")
            logger.info(f"  Columns: {', '.join(df.columns)}")
            
        except Exception as e:
            logger.error(f"✗ Error saving data: {e}")
    
    def get_collection_stats(self, df: pd.DataFrame) -> Dict:
        """Get statistics about collected data"""
        if df.empty:
            return {"total": 0}
        
        stats = {
            'total_tweets': len(df),
            'unique_users': df['user_id'].nunique(),
            'tweets_with_mentions': (df['mentions'] != '').sum(),
            'tweets_with_hashtags': (df['hashtags'] != '').sum(),
            'replies': (df['reply_to'] != '').sum(),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        }
        
        return stats


# Test the collector
if __name__ == "__main__":
    import yaml
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yaml not found!")
        exit(1)
    
    # Test Twitter collector
    logger.info("Testing Twitter Collector...")
    collector = TwitterCollector(config['twitter'])
    
    # Collect data (5 minutes for testing)
    df = collector.collect_real_time(duration_minutes=5)
    
    if not df.empty:
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'data/raw/twitter_test_{timestamp}.csv'
        collector.save_data(df, output_path)
        
        # Show stats
        stats = collector.get_collection_stats(df)
        
        print("\n" + "=" * 80)
        print("COLLECTION STATISTICS")
        print("=" * 80)
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("\n" + "=" * 80)
        print("SAMPLE DATA (first 5 rows)")
        print("=" * 80)
        print(df.head().to_string())
    else:
        logger.warning("No data collected. Check your API credentials and keywords.")