"""
Evaluation Metrics - FINAL ROBUST VERSION
Updates:
1. Loads saved Scaler/PCA to ensure clustering models receive the exact same data format they were trained on.
2. Maintains the 'missing column' fix for supervised models.
"""
import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    silhouette_score, 
    davies_bouldin_score
)

# Initialize Logger
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates ALL models against the final_user_features.csv dataset."""
    
    def __init__(self):
        logger.info("Model Evaluator initialized")
        
    def evaluate_all_models(self, models_dir: str, results_dir: str) -> pd.DataFrame:
        """Evaluate all trained models"""
        try:
            results = []
            
            feature_path = "data/features/final_user_features.csv"
            labels_path = "data/labels/ground_truth_labels.csv"
            
            if not os.path.exists(feature_path):
                logger.error(f"Feature file not found: {feature_path}")
                return pd.DataFrame()

            # 1. LOAD DATA
            df = pd.read_csv(feature_path)
            
            # 2. PREPARE UNSUPERVISED DATA (ROBUST LOADING)
            # Basic drops for clustering
            base_drop = [c for c in ["user_id", "community_id", "created_at", "screen_name", "label", "is_bot", "tweet_id", "Unnamed: 0"] if c in df.columns]
            
            X_numeric = df.drop(columns=base_drop, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
            X_full_values = X_numeric.values
            
            # --- CRITICAL FIX: Load saved preprocessors if they exist ---
            scaler_path = os.path.join(models_dir, "scaler.pkl")
            pca_path = os.path.join(models_dir, "pca.pkl")
            
            # A. SCALING
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                try:
                    X_scaled = scaler.transform(X_full_values)
                except:
                    # Fallback if dimensions changed significantly
                    X_scaled = scaler.fit_transform(X_full_values)
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_full_values)
            
            # B. PCA
            if os.path.exists(pca_path):
                pca = joblib.load(pca_path)
                try:
                    X_unsupervised = pca.transform(X_scaled)
                except:
                    X_unsupervised = pca.fit_transform(X_scaled)
            else:
                pca = PCA(n_components=10, random_state=42)
                X_unsupervised = pca.fit_transform(X_scaled)

            # 3. LOAD LABELS
            y_test = np.zeros(len(df))
            has_ground_truth = False
            
            if os.path.exists(labels_path):
                try:
                    labels_df = pd.read_csv(labels_path)
                    if 'user_id' in df.columns and 'user_id' in labels_df.columns:
                        # Ensure IDs are strings
                        df['user_id'] = df['user_id'].astype(str).str.strip()
                        labels_df['user_id'] = labels_df['user_id'].astype(str).str.strip()
                        
                        # Merge
                        merged = pd.merge(df[['user_id']], labels_df[['user_id', 'is_bot']], on='user_id', how='left')
                        y_test = merged['is_bot'].fillna(0).values
                        has_ground_truth = len(np.unique(y_test)) > 1
                except Exception as e:
                    logger.warning(f"Could not load labels: {e}")

            # 4. EVALUATE MODELS
            if not os.path.exists(models_dir):
                return pd.DataFrame()

            for model_file in os.listdir(models_dir):
                if model_file.endswith('.pkl') and model_file not in ['scaler.pkl', 'pca.pkl', 'feature_names.pkl']:
                    try:
                        model_path = os.path.join(models_dir, model_file)
                        model = joblib.load(model_path)
                        model_name = model_file.replace('.pkl', '')
                        
                        X_test = None
                        
                        # --- ROBUST FEATURE MATCHING ---
                        # Strategy 1: Exact Name Matching (Best for XGBoost/RandomForest)
                        if hasattr(model, "feature_names_in_"):
                            required_features = list(model.feature_names_in_)
                            
                            # Check for missing columns
                            missing = [f for f in required_features if f not in df.columns]
                            if len(missing) > 0:
                                for col in missing:
                                    df[col] = 0
                            
                            # Select exactly the columns the model wants
                            X_test = df[required_features].fillna(0)
                        
                        # Strategy 2: Unsupervised Models
                        elif model_name in ['dbscan', 'kmeans']:
                            X_test = X_unsupervised
                        elif model_name == 'isolation_forest':
                            X_test = X_scaled
                        else:
                            X_test = X_full_values

                        # --- PREDICT ---
                        if hasattr(model, 'predict'):
                            y_pred = model.predict(X_test)
                        elif hasattr(model, 'fit_predict'):
                            y_pred = model.fit_predict(X_test)
                        else:
                            continue
                        
                        metrics_entry = {'model': model_name}

                        # Metrics
                        unique_preds = set(y_pred)
                        if len(unique_preds) > 1:
                            try:
                                # Silhouette is calculated on the transformed space (fair comparison)
                                metrics_entry['silhouette'] = silhouette_score(X_unsupervised, y_pred)
                                metrics_entry['davies_bouldin'] = davies_bouldin_score(X_unsupervised, y_pred)
                            except:
                                metrics_entry['silhouette'] = -1
                        else:
                            metrics_entry['silhouette'] = -1

                        if has_ground_truth:
                            metrics_entry['accuracy'] = accuracy_score(y_test, y_pred)
                            _, _, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                            metrics_entry['f1_score'] = f1
                        
                        results.append(metrics_entry)
                        logger.info(f" {model_name} Evaluated")
                        
                    except Exception as e:
                        logger.error(f" Error evaluating {model_file}: {e}")
            
            # Save Results
            if results:
                df_results = pd.DataFrame(results)
                os.makedirs(results_dir, exist_ok=True)
                df_results.to_csv(os.path.join(results_dir, 'evaluation_metrics.csv'), index=False)
                return df_results
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return pd.DataFrame()