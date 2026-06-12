"""
Community Detection - FIXED VERSION
Detects coordinated communities in graphs
"""

import pandas as pd
import networkx as nx
import logging
from typing import Dict, Optional
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


class CommunityDetector:
    """Detects coordinated communities in graphs"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize community detector"""
        self.config = config or {}
        logger.info("Community Detector initialized")
    
    def detect_communities(self, graph_path: str) -> pd.DataFrame:
        """
        Detect communities from graph file
        
        Args:
            graph_path: Path to graph pickle file
        
        Returns:
            DataFrame with user-community assignments
        """
        try:
            # Load graph
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            
            logger.info("Running community detection algorithms")
            print(f"\n{'='*80}")
            print("COMMUNITY DETECTION")
            print(f"{'='*80}")
            print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # Check if graph is empty or too small
            if G.number_of_nodes() == 0:
                logger.warning("Empty graph, no communities to detect")
                print("⚠️  Empty graph - no communities detected")
                return pd.DataFrame(columns=['user_id', 'community_id', 'community_size'])
            
            if G.number_of_edges() == 0:
                logger.warning("No edges in graph, treating each node as separate community")
                print("⚠️  No edges - each user is their own community")
                # Treat each node as its own community
                community_df = pd.DataFrame([
                    {'user_id': node, 'community_id': idx, 'community_size': 1}
                    for idx, node in enumerate(G.nodes())
                ])
                return community_df
            
            # Convert to undirected for community detection
            if G.is_directed():
                G = G.to_undirected()
                print("Converted to undirected graph for community detection")
            
            # Try different community detection algorithms
            partition = None
            algorithm_used = None
            
            # Try Louvain first (best for most cases)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                algorithm_used = "Louvain"
                print(f"Using {algorithm_used} algorithm")
            except ImportError:
                print("⚠️  python-louvain not installed, trying alternatives...")
            
            # Fallback to Label Propagation
            if partition is None:
                try:
                    communities_generator = nx.community.label_propagation_communities(G)
                    communities_list = list(communities_generator)
                    
                    # Convert to partition format
                    partition = {}
                    for comm_id, comm in enumerate(communities_list):
                        for node in comm:
                            partition[node] = comm_id
                    
                    algorithm_used = "Label Propagation"
                    print(f"Using {algorithm_used} algorithm")
                except Exception as e:
                    logger.error(f"Label Propagation failed: {e}")
            
            # Fallback to Connected Components
            if partition is None:
                components = list(nx.connected_components(G))
                partition = {}
                for comm_id, component in enumerate(components):
                    for node in component:
                        partition[node] = comm_id
                
                algorithm_used = "Connected Components"
                print(f"Using {algorithm_used} algorithm")
            
            if partition is None:
                logger.error("All community detection algorithms failed")
                return pd.DataFrame(columns=['user_id', 'community_id', 'community_size'])
            
            # Convert to DataFrame
            community_df = pd.DataFrame([
                {'user_id': user, 'community_id': comm}
                for user, comm in partition.items()
            ])
            
            # Add community statistics
            community_sizes = community_df.groupby('community_id').size()
            community_df['community_size'] = community_df['community_id'].map(community_sizes)
            
            # Add algorithm info
            community_df['algorithm'] = algorithm_used
            
            print(f"\n📊 Community Detection Results:")
            print(f"   Total communities found: {len(community_sizes)}")
            print(f"   Community size distribution:")
            for comm_id, size in community_sizes.items():
                print(f"      Community {comm_id}: {size} users")
            
            # Filter by minimum size
            min_size = self.config.get('min_community_size', 3)
            before_filter = len(community_df)
            community_df = community_df[community_df['community_size'] >= min_size]
            after_filter = len(community_df)
            
            if before_filter > after_filter:
                print(f"\n🔍 Filtered out communities smaller than {min_size} users")
                print(f"   Users before filter: {before_filter}")
                print(f"   Users after filter: {after_filter}")
            
            # Final statistics
            remaining_communities = community_df['community_id'].nunique()
            logger.info(f"Detected {remaining_communities} communities (min size: {min_size})")
            print(f"\nFinal result: {remaining_communities} communities with ≥{min_size} users")
            print(f"{'='*80}\n")
            
            return community_df
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['user_id', 'community_id', 'community_size'])
    
    def analyze_community_coordination(
        self,
        graph_path: str,
        embeddings_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analyze coordination within communities
        
        Args:
            graph_path: Path to graph file
            embeddings_path: Optional path to BERT embeddings
        
        Returns:
            DataFrame with coordination scores
        """
        try:
            # Load graph
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            
            # Detect communities first
            communities_df = self.detect_communities(graph_path)
            
            if len(communities_df) == 0:
                return communities_df
            
            # Calculate coordination metrics for each community
            coordination_scores = []
            
            for comm_id in communities_df['community_id'].unique():
                comm_users = communities_df[
                    communities_df['community_id'] == comm_id
                ]['user_id'].tolist()
                
                # Get subgraph for this community
                subgraph = G.subgraph(comm_users)
                
                # Calculate metrics
                density = nx.density(subgraph) if len(comm_users) > 1 else 0
                
                coordination_scores.append({
                    'community_id': comm_id,
                    'density': density,
                    'size': len(comm_users)
                })
            
            coordination_df = pd.DataFrame(coordination_scores)
            
            # Merge with communities
            result = communities_df.merge(
                coordination_df[['community_id', 'density']],
                on='community_id',
                how='left'
            )
            
            result.rename(columns={'density': 'coordination_score'}, inplace=True)
            
            logger.info("Coordination analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing coordination: {e}")
            return communities_df


# Test if running directly
if __name__ == "__main__":
    print("Testing Community Detector...")
    print("=" * 80)
    
    # Create test graph
    G = nx.Graph()
    
    # Add bot cluster (highly connected)
    bots = ['bot1', 'bot2', 'bot3', 'bot4', 'bot5']
    for i, bot in enumerate(bots):
        G.add_node(bot)
        for other_bot in bots[i+1:]:
            G.add_edge(bot, other_bot)
    
    # Add human cluster (loosely connected)
    humans = ['human1', 'human2', 'human3']
    for i, human in enumerate(humans):
        G.add_node(human)
        if i < len(humans) - 1:
            G.add_edge(human, humans[i+1])
    
    # Save test graph
    with open('test_graph.gpickle', 'wb') as f:
        pickle.dump(G, f)
    
    # Test detection
    detector = CommunityDetector({'min_community_size': 3})
    communities = detector.detect_communities('test_graph.gpickle')
    
    print("\nDetected communities:")
    print(communities)
    
    print("\nCommunity Detector test complete")