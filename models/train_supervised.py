import pandas as pd
import numpy as np
import logging
import joblib
import os
from typing import Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Initialize Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupervisedDetector:
    """Trains supervised ML models for bot detection"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.models = {}
        self.feature_names = [] # Store feature names here for saving later
        logger.info("Supervised Detector initialized")
    
    def train_models(self):
        """Train all configured models"""
        try:
            # 1. Load and Split Data
            data_split = self._load_data()
            
            # Stop if data loading failed (prevent crash)
            if data_split is None:
                logger.error("Data loading failed. Aborting training.")
                return 
            
            X_train, X_test, y_train, y_test = data_split
            
            # --- CRITICAL FIX 1: Store feature names in the class instance ---
            # We need this to save the list later for the dashboard
            self.feature_names = X_train.columns.tolist()
            logger.info(f"Training with {len(self.feature_names)} features.")

            # 2. DEFINE DEFAULTS
            # If 'models' is missing from config, default to training ALL of them.
            models_to_train = self.config.get('models', ['random_forest', 'xgboost', 'logistic_regression'])
            logger.info(f"Models selected for training: {models_to_train}")

            # 3. Train Random Forest
            if 'random_forest' in models_to_train:
                rf_params = self.config.get('random_forest', {'n_estimators': 100, 'random_state': 42})
                logger.info("Training Random Forest...")
                rf = RandomForestClassifier(**rf_params)
                rf.fit(X_train, y_train)
                self.models['random_forest'] = rf
                logger.info(f"Random Forest Score: {rf.score(X_test, y_test):.4f}")
            
            # 4. Train XGBoost
            if 'xgboost' in models_to_train:
                xgb_params = self.config.get('xgboost', {'use_label_encoder': False, 'eval_metric': 'logloss'})
                logger.info("Training XGBoost...")
                xgb = XGBClassifier(**xgb_params)
                xgb.fit(X_train, y_train)
                self.models['xgboost'] = xgb
                logger.info(f"XGBoost Score: {xgb.score(X_test, y_test):.4f}")
            
            # 5. Train Logistic Regression
            if 'logistic_regression' in models_to_train:
                logger.info("Training Logistic Regression...")
                lr = LogisticRegression(max_iter=1000, random_state=42)
                lr.fit(X_train, y_train)
                self.models['logistic_regression'] = lr
                logger.info(f"Logistic Regression Score: {lr.score(X_test, y_test):.4f}")
            
            # 6. Save Models immediately after training
            self.save_models("data/models")
                
        except Exception as e:
            logger.error(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_data(self):
        """Load features and labels, ensuring IDs match types and preventing leaks"""
        logger.info("Loading training data...")
        
        try:
            # 1. Load Files
            feat_path = "data/features/final_user_features.csv"
            label_path = "data/labels/ground_truth_labels.csv"

            if not os.path.exists(feat_path):
                logger.error(f"Feature file not found: {feat_path}")
                return None
            if not os.path.exists(label_path):
                logger.error(f"Label file not found: {label_path}")
                return None

            X = pd.read_csv(feat_path)
            y = pd.read_csv(label_path)

            # 2. Fix IDs: Force String
            X['user_id'] = X['user_id'].astype(str).str.strip()
            y['user_id'] = y['user_id'].astype(str).str.strip()

            # 3. Merge Data
            df = X.merge(y, on="user_id")
            logger.info(f"Features: {len(X)}, Labels: {len(y)} -> Merged Training Set: {len(df)}")
            
            if len(df) == 0:
                logger.error("CRITICAL: Merge resulted in 0 rows. Check user_ID formats.")
                return None

            # 4. CRITICAL FIX 2: Prevent Data Leakage
            # Remove IDs, labels, and timestamps from the feature set
            leak_cols = [
                "user_id", "is_bot", "label", "community_id", 
                "screen_name", "created_at", "tweet_id", "Unnamed: 0"
            ]
            
            # Optional: Remove highly correlated "easy" features if you suspect overfitting
            # (Uncomment the line below if 100% accuracy persists)
            # leak_cols += ['total_posts', 'statuses_count']
            
            X_data = df.drop(columns=leak_cols, errors='ignore')
            y_data = df["is_bot"] # Target variable
            
            # Ensure X_data is purely numeric
            X_data = X_data.select_dtypes(include=[np.number])

            # 5. Split
            return train_test_split(
                X_data, y_data,
                test_size=0.2,
                random_state=42,
                stratify=y_data
            )

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def save_models(self, output_dir: str):
        """Save trained models and feature names to disk"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Save Models
            for name, model in self.models.items():
                model_path = os.path.join(output_dir, f"{name}.pkl")
                joblib.dump(model, model_path)
                logger.info(f"Saved model: {model_path}")

            # 2. --- CRITICAL FIX 3: Save Feature Names SEPARATELY ---
            # Do NOT overwrite the model file. Save to a distinct file.
            if self.feature_names:
                features_path = os.path.join(output_dir, "feature_names.pkl")
                joblib.dump(self.feature_names, features_path)
                logger.info(f"Saved feature names list to: {features_path}")
            else:
                logger.warning("No feature names found to save.")

        except Exception as e:
            logger.error(f"Error saving models: {e}")

if __name__ == "__main__":
    # Test run
    detector = SupervisedDetector()
    detector.train_models()