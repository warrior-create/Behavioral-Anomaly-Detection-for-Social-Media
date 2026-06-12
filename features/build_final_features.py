import pandas as pd

stylo = pd.read_csv("data/features/test_data_cleaned_stylo.csv")
temporal = pd.read_csv("data/features/test_data_cleaned_temporal.csv")
graph = pd.read_csv("data/features/graph_features.csv")
communities = pd.read_csv(
    "data/results/test_data_cleaned_graph_communities.csv"
)

final = stylo.merge(temporal, on="user_id", how="left")
final = final.merge(graph, on="user_id", how="left")
final = final.merge(
    communities[["user_id", "coordination_score", "community_size"]],
    on="user_id",
    how="left"
)

final.fillna(0, inplace=True)
final.to_csv("data/features/final_user_features.csv", index=False)

print("✅ FINAL ML FEATURE TABLE CREATED:", final.shape)