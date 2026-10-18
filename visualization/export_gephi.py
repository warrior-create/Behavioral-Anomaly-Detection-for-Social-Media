# ==============================================================================
# GEPHI EXPORTER - visualization/gephi_exporter.py
# ==============================================================================
import networkx as nx
import logging
import pickle
import os
from pathlib import Path
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

class GephiExporter:
    """Exports graphs to Gephi-compatible GEXF format"""
    
    def __init__(self):
        logger.info("Gephi Exporter initialized")
    
    def export_graph(self, graph_path: str, output_path: str, communities_path: Optional[str] = None):
        """Export graph to GEXF format for Gephi"""
        try:
            if not os.path.exists(graph_path):
                logger.error(f"Graph file not found: {graph_path}")
                return

            logger.info(f"Loading graph from {graph_path}")
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            
            # Add Attributes (Degree, Community, etc)
            self._add_node_attributes(G, communities_path)
            
            # Ensure Output Directory Exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Export
            nx.write_gexf(G, output_path)
            logger.info(f"✅ Graph exported successfully to {output_path}")
            
            # Create Guide
            self.create_visualization_guide(os.path.dirname(output_path))
            
        except Exception as e:
            logger.error(f"Error exporting graph: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_node_attributes(self, G: nx.Graph, communities_path: Optional[str] = None):
        """Add node attributes (Degree, Community) for visualization"""
        # Load community data
        community_map = {}
        if communities_path and os.path.exists(communities_path):
            try:
                comm_df = pd.read_csv(communities_path)
                # Assuming columns are user_id and community_id
                if 'user_id' in comm_df.columns and 'community_id' in comm_df.columns:
                    community_map = dict(zip(comm_df['user_id'], comm_df['community_id']))
            except Exception:
                pass
        
        for node in G.nodes():
            # 1. Degree
            G.nodes[node]['degree'] = G.degree(node)
            
            # 2. Label
            G.nodes[node]['label'] = str(node)
            
            # 3. Community (Color)
            if node in community_map:
                G.nodes[node]['modularity_class'] = int(community_map[node])
            else:
                G.nodes[node]['modularity_class'] = 0

    def create_visualization_guide(self, output_dir: str):
        """Create a text file with instructions for Gephi"""
        guide_path = os.path.join(output_dir, "GEPHI_GUIDE.txt")
        content = """
============================================================
HOW TO VISUALIZE THIS GRAPH IN GEPHI
============================================================

1. Download Gephi from https://gephi.org/users/download/
2. Open Gephi and select "Open Graph File"
3. Select the .gexf file in this folder.
4. In the Import Report window, leave settings as default and click OK.

------------------------------------------------------------
Recommended Visualization Steps:
------------------------------------------------------------
A. LAYOUT (Left Panel):
   1. Select "Force Atlas 2"
   2. Check "Dissuade Hubs" and "Prevent Overlap"
   3. Click Run and wait for the graph to stabilize.

B. APPEARANCE (Left Panel - Top):
   1. Click the "Nodes" tab -> "Partition" (Palette Icon)
   2. Choose "modularity_class" from the dropdown.
   3. Click Apply. (This colors nodes by Community/Bot Group)
   
   4. Click the "Ranking" (Size Icon)
   5. Choose "Degree"
   6. Set Min size 10, Max size 50. Click Apply.

C. STATISTICS (Right Panel):
   1. Run "Network Diameter" to see the graph structure.
   2. Run "Modularity" to detect new communities if needed.
============================================================
        """
        try:
            with open(guide_path, 'w') as f:
                f.write(content)
        except Exception:
            pass