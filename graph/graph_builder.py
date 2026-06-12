"""
Graph Builder - FINAL FIXED VERSION
Builds interaction graphs from social media data.
Fixes applied:
1. Strips '@' from mentions.
2. Handles list parsing properly.
3. Forces string IDs.
4. Includes all required statistics methods.
"""

import pandas as pd
import networkx as nx
import logging
from typing import Dict, List, Optional
import pickle
import ast

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Builds interaction graphs from social media data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize graph builder"""
        self.config = config or {}
        logger.info("Graph Builder initialized")
    
    def build_graph(self, file_path: str) -> nx.Graph:
        """Build interaction graph from data file"""
        try:
            # 1. Load Data
            df = pd.read_csv(file_path)
            
            # --- CRITICAL DATA CLEANING ---
            # Ensure user_ids are strings to avoid int/str mismatches
            df['user_id'] = df['user_id'].astype(str).str.strip()
            
            # Ensure post_ids are strings and remove float decimals if present
            df['post_id'] = df['post_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            # ------------------------------
            
            logger.info(f"Building graph from {len(df)} posts")
            
            # Initialize directed/undirected graph
            if self.config.get('directed', True):
                G = nx.DiGraph()
            else:
                G = nx.Graph()
            
            # Add all users as nodes first
            users = set(df['user_id'].unique())
            G.add_nodes_from(users)
            logger.info(f"Added {len(users)} user nodes")
            
            # Add edges based on interactions
            edge_types = self.config.get('edge_types', ['mention', 'reply'])
            edge_count = 0
            
            if 'mention' in edge_types:
                mentions_added = self._add_mention_edges(G, df)
                edge_count += mentions_added
                logger.info(f"Added {mentions_added} mention edges")
            
            if 'reply' in edge_types:
                replies_added = self._add_reply_edges(G, df)
                edge_count += replies_added
                logger.info(f"Added {replies_added} reply edges")
            
            logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            import traceback
            traceback.print_exc()
            return nx.Graph()
    
    def _add_mention_edges(self, G: nx.Graph, df: pd.DataFrame) -> int:
        """Add edges for mentions - Includes '@' stripping fix"""
        edges_added = 0
        
        for _, row in df.iterrows():
            user = row['user_id']
            
            if 'mentions' in df.columns and pd.notna(row['mentions']):
                mentions_val = str(row['mentions']).strip()
                
                # Skip empty brackets
                if mentions_val == '[]' or mentions_val == '':
                    continue
                
                mentions = []
                
                # --- PARSING LOGIC ---
                try:
                    # Check if it looks like a list "['val']"
                    if mentions_val.startswith('['):
                        mentions = ast.literal_eval(mentions_val)
                    else:
                        # Fallback for simple comma separation
                        mentions = mentions_val.split(',')
                except:
                    # Final fallback cleanup
                    mentions = [m.strip(" '\"[]") for m in mentions_val.split(',')]
                # ---------------------

                for mentioned in mentions:
                    # --- THE FIX: Strip '@' and whitespace ---
                    mentioned = str(mentioned).strip().lstrip('@')
                    
                    if mentioned and mentioned != user and mentioned in G.nodes():
                        if G.has_edge(user, mentioned):
                            G[user][mentioned]['weight'] += 1
                        else:
                            G.add_edge(user, mentioned, weight=1, type='mention')
                            edges_added += 1
        
        return edges_added
    
    def _add_reply_edges(self, G: nx.Graph, df: pd.DataFrame) -> int:
        """Add edges for replies"""
        edges_added = 0
        
        # Create mapping from post_id to user_id (using clean strings)
        post_to_user = dict(zip(df['post_id'], df['user_id']))
        
        for _, row in df.iterrows():
            user = row['user_id']
            
            if 'reply_to' in df.columns and pd.notna(row['reply_to']):
                # Clean the reply_to ID
                replied_post = str(row['reply_to']).replace('.0', '').strip()
                
                # Find who posted the original
                if replied_post in post_to_user:
                    replied_user = post_to_user[replied_post]
                    
                    if replied_user != user and replied_user in G.nodes():
                        if G.has_edge(user, replied_user):
                            G[user][replied_user]['weight'] += 1
                        else:
                            G.add_edge(user, replied_user, weight=1, type='reply')
                            edges_added += 1
        
        return edges_added
    
    def get_graph_statistics(self, G: nx.Graph) -> Dict:
        """Compute graph statistics"""
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': 0,
            'avg_degree': 0
        }
        
        if G.number_of_nodes() > 0:
            stats['density'] = nx.density(G)
            degrees = dict(G.degree())
            stats['avg_degree'] = sum(degrees.values()) / len(degrees)
            
            if G.number_of_edges() > 0:
                stats['max_degree'] = max(degrees.values())
                stats['min_degree'] = min(degrees.values())
                
                # Connected components (for undirected view)
                G_undirected = G.to_undirected()
                stats['num_components'] = nx.number_connected_components(G_undirected)
                
                # Get largest component size
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                stats['largest_component_size'] = len(largest_cc)
        
        return stats
    
    def save_graph(self, G: nx.Graph, output_path: str):
        """Save graph to pickle file"""
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(G, f)
            logger.info(f"Graph saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    def load_graph(self, file_path: str) -> nx.Graph:
        """Load graph from pickle file"""
        try:
            with open(file_path, 'rb') as f:
                G = pickle.load(f)
            logger.info(f"Loaded graph from {file_path}")
            return G
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return nx.Graph()
    
    def visualize_graph_summary(self, G: nx.Graph):
        """Print graph summary"""
        print("\n" + "=" * 80)
        print("GRAPH SUMMARY")
        print("=" * 80)
        
        stats = self.get_graph_statistics(G)
        
        print(f"Nodes: {stats['num_nodes']}")
        print(f"Edges: {stats['num_edges']}")
        print(f"Density: {stats['density']:.6f}")
        print(f"Average Degree: {stats['avg_degree']:.4f}")
        
        if 'num_components' in stats:
            print(f"Connected Components: {stats['num_components']}")
            print(f"Largest Component: {stats['largest_component_size']} nodes")
        
        print("=" * 80 + "\n")