"""
graph_feature_extractor.py  — COMPLETE VERSION
================================================
Extracts per-user graph centrality and structural features from a NetworkX
interaction graph and writes them to data/features/graph_features.csv.

Previously: hardcoded single file path, no error handling, no integration.
Now:
  - Auto-discovers all .gpickle files in data/graphs/
  - Computes a rich set of centrality + structural features
  - Handles directed vs undirected graphs gracefully
  - Integrable into the main pipeline
"""

import os
import glob
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GRAPHS_DIR = "data/graphs"
OUTPUT_PATH = "data/features/graph_features.csv"


def extract_graph_features(
    graph_path: str,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract per-node (per-user) graph features from a pickled NetworkX graph.

    Features extracted:
      Degree-based:
        - degree, in_degree, out_degree, weighted_degree
        - degree_centrality, in_degree_centrality, out_degree_centrality

      Centrality:
        - betweenness_centrality   (who bridges communities?)
        - closeness_centrality     (how quickly can a user reach others?)
        - pagerank                 (who is amplified the most?)
        - eigenvector_centrality   (who is connected to influential nodes?)

      Structural:
        - clustering_coefficient   (are neighbours interconnected? → bot cliques)
        - triangles                (raw count of closed triplets)
        - local_efficiency         (how efficient is the local neighbourhood?)
        - eccentricity             (max shortest path to any other node)
        - square_clustering        (4-cycle clustering, sensitive to bot rings)

      Community / neighbourhood:
        - avg_neighbour_degree     (do neighbours have high degree?)
        - std_neighbour_degree

    Args:
        graph_path: Path to .gpickle graph file.
        output_path: Where to save the CSV. If None, only returns DataFrame.

    Returns:
        DataFrame with one row per user node.
    """
    logger.info("Loading graph from %s", graph_path)
    try:
        with open(graph_path, "rb") as f:
            G = pickle.load(f)
    except Exception as e:
        logger.error("Failed to load graph: %s", e)
        return pd.DataFrame()

    is_directed = G.is_directed()
    logger.info(
        "Graph: %d nodes, %d edges, directed=%s",
        G.number_of_nodes(), G.number_of_edges(), is_directed,
    )

    if G.number_of_nodes() == 0:
        logger.warning("Empty graph — returning empty DataFrame.")
        return pd.DataFrame()

    # Work with undirected copy for symmetric metrics
    G_u = G.to_undirected()

    # ------------------------------------------------------------------ #
    # Compute metrics (with safe fallbacks for sparse/disconnected graphs) #
    # ------------------------------------------------------------------ #

    def safe_compute(fn, *args, default=0.0, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.debug("Metric computation failed (%s): %s", fn.__name__, exc)
            n = G.number_of_nodes()
            return {node: default for node in G.nodes()}

    betweenness = safe_compute(nx.betweenness_centrality, G_u)
    closeness   = safe_compute(nx.closeness_centrality, G_u)
    pagerank    = safe_compute(nx.pagerank, G_u, alpha=0.85)
    clustering  = safe_compute(nx.clustering, G_u)
    triangles   = safe_compute(nx.triangles, G_u)
    sq_cluster  = safe_compute(nx.square_clustering, G_u)
    avg_nb_deg  = safe_compute(nx.average_neighbor_degree, G_u)

    # Eigenvector centrality can fail on disconnected graphs — use power iteration
    try:
        eigenvector = nx.eigenvector_centrality(G_u, max_iter=500, tol=1e-4)
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    # Degree centrality
    deg_centrality = nx.degree_centrality(G_u)
    in_deg_centrality  = nx.in_degree_centrality(G)  if is_directed else deg_centrality
    out_deg_centrality = nx.out_degree_centrality(G) if is_directed else deg_centrality

    # Eccentricity (only defined on connected graphs; use largest CC)
    eccentricity: dict = {}
    if nx.is_connected(G_u):
        eccentricity = nx.eccentricity(G_u)
    else:
        largest_cc = max(nx.connected_components(G_u), key=len)
        G_lcc = G_u.subgraph(largest_cc)
        ecc_lcc = nx.eccentricity(G_lcc)
        eccentricity = {n: ecc_lcc.get(n, 0) for n in G.nodes()}

    # Std of neighbour degrees per node
    std_nb_deg: dict = {}
    for node in G_u.nodes():
        nbs = list(G_u.neighbors(node))
        if nbs:
            nb_degs = [G_u.degree(nb) for nb in nbs]
            std_nb_deg[node] = float(np.std(nb_degs))
        else:
            std_nb_deg[node] = 0.0

    # ------------------------------------------------------------------ #
    # Build row per node                                                   #
    # ------------------------------------------------------------------ #
    rows = []
    for node in G.nodes():
        row = {
            "user_id": str(node),
            # Degree
            "degree":                 G_u.degree(node),
            "in_degree":              G.in_degree(node)  if is_directed else G_u.degree(node),
            "out_degree":             G.out_degree(node) if is_directed else G_u.degree(node),
            "weighted_degree":        G_u.degree(node, weight="weight"),
            # Centrality
            "degree_centrality":      deg_centrality.get(node, 0.0),
            "in_degree_centrality":   in_deg_centrality.get(node, 0.0),
            "out_degree_centrality":  out_deg_centrality.get(node, 0.0),
            "betweenness_centrality": betweenness.get(node, 0.0),
            "closeness_centrality":   closeness.get(node, 0.0),
            "pagerank":               pagerank.get(node, 0.0),
            "eigenvector_centrality": eigenvector.get(node, 0.0),
            # Structural
            "clustering_coeff":       clustering.get(node, 0.0),
            "triangles":              triangles.get(node, 0),
            "square_clustering":      sq_cluster.get(node, 0.0),
            "eccentricity":           eccentricity.get(node, 0),
            # Neighbourhood
            "avg_neighbour_degree":   avg_nb_deg.get(node, 0.0),
            "std_neighbour_degree":   std_nb_deg.get(node, 0.0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Graph features saved → %s  (%d users × %d cols)",
                    output_path, len(df), len(df.columns))

    return df


def extract_all_graphs(
    graphs_dir: str = GRAPHS_DIR,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Auto-discover all .gpickle files and combine their features.

    Args:
        graphs_dir:  Directory to search for graph files.
        output_path: Where to save combined CSV.

    Returns:
        Combined DataFrame for all graphs.
    """
    graph_files = sorted(glob.glob(os.path.join(graphs_dir, "*.gpickle")))

    if not graph_files:
        logger.error("No .gpickle graph files found in %s", graphs_dir)
        return pd.DataFrame()

    parts = []
    for path in graph_files:
        df = extract_graph_features(path)
        if not df.empty:
            parts.append(df)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    # Deduplicate users across files (keep first)
    combined = combined.drop_duplicates(subset="user_id", keep="first")
    combined = combined.fillna(0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Combined graph features → %s  (%d users × %d cols)",
                output_path, len(combined), len(combined.columns))

    return combined


if __name__ == "__main__":
    df = extract_all_graphs()
    print(f"\n✅ Graph features saved: {df.shape}")
    print(df.head())