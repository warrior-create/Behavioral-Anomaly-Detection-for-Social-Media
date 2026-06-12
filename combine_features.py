import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_feature_files():
    # Define file paths based on your screenshot
    base_dir = "data/features"
    file_paths = {
        "graph": os.path.join(base_dir, "graph_features.csv"),
        "stylo": os.path.join(base_dir, "test_data_cleaned_stylo.csv"),
        "temporal": os.path.join(base_dir, "test_data_cleaned_temporal.csv")
    }
    output_path = os.path.join(base_dir, "final_user_features.csv")

    dfs = []
    
    # 1. Load each file
    for name, path in file_paths.items():
        if os.path.exists(path):
            logger.info(f"Loading {name} features from {path}...")
            df = pd.read_csv(path)
            
            # CRITICAL: Force User IDs to string to ensure matching works
            if 'user_id' in df.columns:
                df['user_id'] = df['user_id'].astype(str).str.strip()
                dfs.append(df)
            else:
                logger.warning(f"⚠️ Skipping {name}: 'user_id' column missing.")
        else:
            logger.error(f"❌ File not found: {path}")
            return

    # 2. Merge them all on 'user_id'
    if not dfs:
        logger.error("No valid dataframes to merge.")
        return

    logger.info("Merging files...")
    # Start with the first dataframe
    final_df = dfs[0]
    
    # Merge the rest (using outer join to keep as much data as possible, or inner for strict matching)
    for i in range(1, len(dfs)):
        final_df = pd.merge(final_df, dfs[i], on='user_id', how='inner')
    
    # 3. Fill NaNs (important for ML models)
    final_df = final_df.fillna(0)

    # 4. Save
    final_df.to_csv(output_path, index=False)
    logger.info(f" Successfully created combined file at: {output_path}")
    logger.info(f"Final Shape: {final_df.shape} (Rows, Columns)")

if __name__ == "__main__":
    combine_feature_files()