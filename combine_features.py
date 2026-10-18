"""
combine_features.py  — COMPLETE VERSION
========================================
Merges ALL feature files (stylometric, temporal, graph, community,
TF-IDF, NER, LDA topics, perplexity, PMI) into a single
final_user_features.csv for model training.

Previously: hardcoded filenames, only 3 feature sources.
Now: auto-discovers all feature CSVs by pattern, includes all NLP modules.
"""

import os
import glob
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Suffix → display-name mapping for each feature group
# -----------------------------------------------------------------------
FEATURE_GROUPS = {
    "_stylo.csv":        "Stylometric",
    "_temporal.csv":     "Temporal",
    "_tfidf.csv":        "TF-IDF",
    "_ner.csv":          "NER",
    "_topics.csv":       "LDA Topics",
    "_perplexity.csv":   "Perplexity",
    "_pmi.csv":          "PMI Co-occurrence",
}

# Graph features are stored with a fixed name (produced by graph_feature_extractor)
GRAPH_FEATURE_FILE = "data/features/graph_features.csv"
COMMUNITY_RESULT_GLOB = "data/results/*_communities.csv"
OUTPUT_PATH = "data/features/final_user_features.csv"


def _load_and_cast(path: str, label: str) -> pd.DataFrame:
    """Load a CSV, cast user_id to str, log shape."""
    df = pd.read_csv(path)
    if "user_id" not in df.columns:
        logger.warning("'user_id' missing in %s — skipping.", path)
        return pd.DataFrame()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    logger.info("  ✓ %s: %d rows × %d cols", label, len(df), len(df.columns))
    return df


def _discover_stem_features(base_dir: str) -> dict:
    """
    Discover per-stem feature files.

    For each data file stem (e.g. 'test_data_cleaned') find all matching
    <stem><suffix> CSV files and return them grouped by stem.

    Returns: {stem: {group_name: DataFrame}}
    """
    stem_dfs: dict = {}

    for suffix, group_name in FEATURE_GROUPS.items():
        pattern = os.path.join(base_dir, f"*{suffix}")
        for path in sorted(glob.glob(pattern)):
            stem = os.path.basename(path).replace(suffix, "")
            df = _load_and_cast(path, f"{group_name} [{stem}]")
            if df.empty:
                continue
            if stem not in stem_dfs:
                stem_dfs[stem] = {}
            stem_dfs[stem][group_name] = df

    return stem_dfs


def _merge_stem_features(stem_dfs: dict) -> pd.DataFrame:
    """
    Merge all per-stem feature DataFrames on user_id using outer joins,
    then concatenate across stems.
    """
    all_merged = []

    for stem, group_map in stem_dfs.items():
        logger.info("Merging groups for stem: %s", stem)
        merged = None
        for group_name, df in group_map.items():
            if merged is None:
                merged = df
            else:
                # Outer join so no users are dropped if one feature group
                # has slightly different user coverage
                merged = pd.merge(merged, df, on="user_id", how="outer",
                                  suffixes=("", f"__{group_name}"))
        if merged is not None:
            all_merged.append(merged)

    if not all_merged:
        return pd.DataFrame()

    # Concatenate across different data stems (e.g. multiple input files)
    combined = pd.concat(all_merged, ignore_index=True)
    # If a user_id appears multiple times (across stems), keep first occurrence
    combined = combined.drop_duplicates(subset="user_id", keep="first")
    return combined


def _attach_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach graph centrality features if available."""
    if not os.path.exists(GRAPH_FEATURE_FILE):
        logger.warning("Graph features not found: %s", GRAPH_FEATURE_FILE)
        return df
    gdf = _load_and_cast(GRAPH_FEATURE_FILE, "Graph Centrality")
    if gdf.empty:
        return df
    df = pd.merge(df, gdf, on="user_id", how="left")
    logger.info("  ✓ Graph features attached.")
    return df


def _attach_community_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach community detection results if available."""
    comm_files = sorted(glob.glob(COMMUNITY_RESULT_GLOB))
    if not comm_files:
        logger.warning("No community result files found matching: %s", COMMUNITY_RESULT_GLOB)
        return df

    comm_parts = []
    for path in comm_files:
        cdf = _load_and_cast(path, f"Community [{os.path.basename(path)}]")
        if not cdf.empty:
            # Keep only the useful per-user columns
            keep_cols = ["user_id"]
            for col in ["community_id", "community_size", "coordination_score"]:
                if col in cdf.columns:
                    keep_cols.append(col)
            comm_parts.append(cdf[keep_cols])

    if not comm_parts:
        return df

    comm_all = pd.concat(comm_parts, ignore_index=True)
    comm_all = comm_all.drop_duplicates(subset="user_id", keep="first")
    df = pd.merge(df, comm_all, on="user_id", how="left")
    logger.info("  ✓ Community features attached.")
    return df


def combine_feature_files(
    base_dir: str = "data/features",
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Main function: discover, merge, and save all feature files.

    Args:
        base_dir:     Directory containing feature CSV files.
        output_path:  Where to write the final combined CSV.

    Returns:
        The final combined DataFrame.
    """
    logger.info("=" * 70)
    logger.info("  Feature Combiner — discovering all feature groups")
    logger.info("=" * 70)

    # 1. Per-stem NLP/stylometry/temporal/etc. features
    stem_dfs = _discover_stem_features(base_dir)

    if not stem_dfs:
        logger.error("No feature files found in %s. Run the pipeline first.", base_dir)
        return pd.DataFrame()

    final_df = _merge_stem_features(stem_dfs)
    logger.info("After NLP feature merge: %d users × %d cols", len(final_df), len(final_df.columns))

    # 2. Graph centrality features
    final_df = _attach_graph_features(final_df)

    # 3. Community membership features
    final_df = _attach_community_features(final_df)

    # 4. Fill NaN with 0 (safe default for numerical features)
    final_df = final_df.fillna(0)

    # 5. Remove duplicate columns (can arise from outer joins with suffixes)
    dupes = [c for c in final_df.columns if "__" in c]
    if dupes:
        logger.info("Removing %d duplicate-suffix columns.", len(dupes))
        final_df.drop(columns=dupes, inplace=True)

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)

    logger.info("=" * 70)
    logger.info("  ✅ Final feature table saved → %s", output_path)
    logger.info("  Shape: %d users × %d features", len(final_df), len(final_df.columns) - 1)
    logger.info("=" * 70)

    # Print feature group breakdown
    _print_feature_summary(final_df)

    return final_df


def _print_feature_summary(df: pd.DataFrame):
    """Print how many columns come from each feature group."""
    print("\n📊 Feature Group Summary:")
    print(f"{'Group':<25} {'# Columns':>10}")
    print("-" * 37)

    group_patterns = {
        "Stylometric": ["char_count", "word_count", "vocab", "hapax", "punct",
                        "upper", "sentiment", "lexical", "function", "repetition"],
        "Temporal": ["inter_post", "burst", "posting", "night", "weekend", "regularity"],
        "TF-IDF": ["tfidf_"],
        "NER": ["ner_"],
        "LDA Topics": ["lda_"],
        "Perplexity": ["perplexity_"],
        "PMI Co-occ": ["pmi_"],
        "Graph": ["degree", "betweenness", "closeness", "pagerank", "clustering"],
        "Community": ["community_id", "community_size", "coordination_score"],
    }

    for group, patterns in group_patterns.items():
        cols = [c for c in df.columns
                if any(p in c for p in patterns) and c != "user_id"]
        if cols:
            print(f"  {group:<23} {len(cols):>10}")

    total_feat = len([c for c in df.columns if c != "user_id"])
    print("-" * 37)
    print(f"  {'TOTAL':<23} {total_feat:>10}")
    print()


if __name__ == "__main__":
    combine_feature_files()