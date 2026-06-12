import pickle
import pandas as pd
import networkx as nx

GRAPH_PATH = "data/graphs/test_data_cleaned_graph.gpickle"
OUTPUT_PATH = "data/features/graph_features.csv"

with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)

G_u = G.to_undirected()

betweenness = nx.betweenness_centrality(G_u)
closeness = nx.closeness_centrality(G_u)
pagerank = nx.pagerank(G_u)
clustering = nx.clustering(G_u)
triangles = nx.triangles(G_u)

rows = []
for node in G.nodes():
    rows.append({
        "user_id": node,
        "degree": G.degree(node),
        "weighted_degree": G.degree(node, weight="weight"),
        "in_degree": G.in_degree(node) if G.is_directed() else G.degree(node),
        "out_degree": G.out_degree(node) if G.is_directed() else G.degree(node),
        "betweenness": betweenness[node],
        "closeness": closeness[node],
        "pagerank": pagerank[node],
        "clustering_coeff": clustering[node],
        "triangles": triangles[node],
    })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(" Graph features saved:", OUTPUT_PATH)