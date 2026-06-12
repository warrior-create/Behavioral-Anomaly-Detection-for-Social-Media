import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, Dict

# Initialize Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnsupervisedDetector:
    """Trains unsupervised models for anomaly detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
        # We must store these to save them later
        self.scaler = None 
        self.pca = None
        logger.info("Unsupervised Detector initialized")
    
    def train_models(self):
        """Train unsupervised models"""
        try:
            # 1. Load Raw Data
            X_raw = self._load_data()
            
            if X_raw is None or len(X_raw) == 0:
                logger.error("No data available for training.")
                return

            logger.info(f"Training models on raw data shape: {X_raw.shape}")

            # 2. Preprocessing (CRITICAL: Must be saved later)
            # Step A: Scale Data
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_raw)
            
            # Step B: Reduce Dimensions (120 -> 10)
            self.pca = PCA(n_components=10, random_state=42)
            X_pca = self.pca.fit_transform(X_scaled)
            
            logger.info(f"Data scaled and reduced to {X_pca.shape} for clustering")

            model_list = self.config.get('models', ['dbscan', 'isolation_forest', 'kmeans'])
            
            # --- DBSCAN ---
            if 'dbscan' in model_list:
                # fit_predict is used because DBSCAN doesn't strictly "predict" on new data easily
                params = self.config.get('dbscan', {'eps': 2.5, 'min_samples': 3})
                dbscan = DBSCAN(**params)
                labels = dbscan.fit_predict(X_pca)
                self.models['dbscan'] = dbscan
                
                # Metric logging
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                logger.info(f"DBSCAN trained - Found {n_clusters} clusters (Noise: {n_noise})")
            
            # --- Isolation Forest ---
            if 'isolation_forest' in model_list:
                # Isolation Forest often works better on the full scaled data (not PCA)
                params = self.config.get('isolation_forest', {'contamination': 0.1, 'random_state': 42})
                iso = IsolationForest(**params)
                iso.fit(X_scaled)
                self.models['isolation_forest'] = iso
                logger.info("Isolation Forest trained")
            
            # --- KMeans ---
            if 'kmeans' in model_list:
                params = self.config.get('kmeans', {'n_clusters': 2, 'random_state': 42})
                kmeans = KMeans(**params)
                kmeans.fit(X_pca)
                self.models['kmeans'] = kmeans
                logger.info(f"KMeans trained - {kmeans.n_clusters} clusters")
            
            # 3. CRITICAL FIX: Save the models AND the preprocessors
            self.save_models("data/models")
                
        except Exception as e:
            logger.error(f"Error training unsupervised models: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_data(self):
        """Load feature data strictly from final_user_features.csv"""
        try:
            feature_path = 'data/features/final_user_features.csv'
            
            if os.path.exists(feature_path):
                logger.info(f"Loading features from {feature_path}")
                df = pd.read_csv(feature_path)
                
                # Drop non-numeric and leakage columns
                # We drop 'label' and 'is_bot' to ensure unsupervised learning is truly unsupervised
                drop_cols = [
                    "user_id", "community_id", "created_at", "screen_name", 
                    "label", "is_bot", "tweet_id", "Unnamed: 0"
                ]
                
                numeric_df = df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
                return numeric_df.fillna(0).values
            
            logger.error(f"Feature file not found: {feature_path}")
            return None

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def save_models(self, output_dir: str):
        """Save trained models AND PREPROCESSORS"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Save Models
            for name, model in self.models.items():
                save_path = os.path.join(output_dir, f"{name}.pkl")
                joblib.dump(model, save_path)
                logger.info(f"Saved model: {save_path}")
            
            # 2. CRITICAL FIX: Save Scaler and PCA
            # Without these, you cannot reproduce the input data format for prediction
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
                logger.info("Saved StandardScaler (scaler.pkl)")
                
            if self.pca:
                joblib.dump(self.pca, os.path.join(output_dir, "pca.pkl"))
                logger.info("Saved PCA (pca.pkl)")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

if __name__ == "__main__":
    detector = UnsupervisedDetector()
    detector.train_models()