"""
build_final_features.py — Compatibility wrapper around combine_features.py
"""
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from combine_features import combine_feature_files

if __name__ == "__main__":
    final_df = combine_feature_files()
    print("✅ FINAL ML FEATURE TABLE CREATED:", final_df.shape)