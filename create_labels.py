"""
Create Labels - HYBRID VERSION
Generates ground truth labels by checking BOTH the label column and the user_id.
"""
import pandas as pd
import os

def create_robust_labels(input_path: str = None, output_path: str = "data/labels/ground_truth_labels.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if input_path is not None and os.path.exists(input_path):
        files = [input_path]
    else:
        import glob
        files = sorted(glob.glob("data/processed/*_cleaned.csv"))
        if not files:
            files = ["data/processed/test_data_cleaned.csv"]

    all_dfs = []
    for f in files:
        if os.path.exists(f):
            print(f"Reading data from: {f}")
            df = pd.read_csv(f)
            df['user_id'] = df['user_id'].astype(str).str.strip()
            all_dfs.append(df)

    if not all_dfs:
        print("❌ Error: No cleaned data files found.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    users = combined_df.groupby('user_id').first().reset_index()
    print(f"Processing {len(users)} unique users...")

    # --- HYBRID LABELING LOGIC ---
    def determine_label(row):
        # 1. Check the explicit 'label' column
        label_val = str(row.get('label', '')).lower()
        if 'bot' in label_val or label_val in ['1', '1.0', 'true']:
            return 1
            
        # 2. Fallback: Check if 'bot' is in the username (heuristic)
        # This catches 'prom_bot_2199', 'cryp_bot_6523', etc.
        user_id_val = str(row['user_id']).lower()
        if 'bot' in user_id_val:
            return 1
            
        return 0

    # Apply logic
    users['is_bot'] = users.apply(determine_label, axis=1)
    
    # Save results
    final_labels = users[['user_id', 'is_bot']]
    final_labels.to_csv(output_path, index=False)
    
    # --- VERIFICATION ---
    print(f"✅ Labels saved to {output_path}")
    print("\n📊 Class Distribution:")
    counts = final_labels['is_bot'].value_counts()
    print(counts)
    
    if 1 in counts:
        print(f"   Human (0): {counts.get(0, 0)}")
        print(f"   Bot   (1): {counts.get(1, 0)}")
        print(f"   Bot Ratio: {counts.get(1, 0) / len(final_labels):.2%}")
    else:
        print("❌ WARNING: Still found 0 bots! Check your data source.")

if __name__ == "__main__":
    create_robust_labels()