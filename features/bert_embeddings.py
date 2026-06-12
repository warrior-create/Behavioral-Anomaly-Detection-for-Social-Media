"""
BERT Embedding Generator
Creates semantic embeddings using sentence-transformers
"""

import pandas as pd
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class BERTEmbedder:
    """Generates BERT embeddings for text similarity analysis"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize BERT embedder
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model_name = self.config.get(
            'model_name',
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        self.device = self._setup_device()
        self.model = self._load_model()
        self.batch_size = self.config.get('batch_size', 32)
        
        logger.info(f"BERT Embedder initialized with {self.model_name} on {self.device}")
    
    def _setup_device(self) -> str:
        """Determine device (CPU/CUDA) for computation"""
        device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
        
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        return device
    
    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_embeddings(self, file_path: str) -> np.ndarray:
        """
        Generate embeddings for texts in CSV file
        
        Args:
            file_path: Path to CSV with 'text' column
        
        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        try:
            df = pd.read_csv(file_path)
            texts = df['text'].fillna('').tolist()
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.encode_texts(texts)
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode list of texts to embeddings
        
        Args:
            texts: List of text strings
        
        Returns:
            Numpy array of embeddings
        """
        try:
            # Handle empty or None texts
            texts = [str(text) if text else "" for text in texts]
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.config.get('normalize_embeddings', True)
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            return np.array([])
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix
        
        Args:
            embeddings: Array of embeddings (n_samples, embedding_dim)
        
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            similarity_matrix = cosine_similarity(embeddings)
            logger.info(f"Computed similarity matrix: {similarity_matrix.shape}")
            
            return similarity_matrix
            
        except Exception as e:
            logger.error(f"Error computing similarity matrix: {e}")
            return np.array([])
    
    def find_similar_texts(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find most similar texts to a query
        
        Args:
            query_embedding: Query embedding vector
            corpus_embeddings: Corpus embedding matrix
            top_k: Number of top results to return
        
        Returns:
            List of (index, similarity_score) tuples
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1),
                corpus_embeddings
            )[0]
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [(idx, similarities[idx]) for idx in top_indices]
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar texts: {e}")
            return []
    
    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 10
    ) -> np.ndarray:
        """
        Cluster embeddings using K-means
        
        Args:
            embeddings: Embedding matrix
            n_clusters: Number of clusters
        
        Returns:
            Cluster labels array
        """
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            logger.info(f"Clustered into {n_clusters} groups")
            return labels
            
        except Exception as e:
            logger.error(f"Error clustering embeddings: {e}")
            return np.array([])
    
    def reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = 'umap',
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization
        
        Args:
            embeddings: High-dimensional embeddings
            method: 'umap', 'tsne', or 'pca'
            n_components: Target dimensionality
        
        Returns:
            Reduced embeddings
        """
        try:
            if method == 'umap':
                from umap import UMAP
                reducer = UMAP(n_components=n_components, random_state=42)
            elif method == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42)
            elif method == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            reduced = reducer.fit_transform(embeddings)
            logger.info(f"Reduced dimensions using {method}: {reduced.shape}")
            
            return reduced
            
        except Exception as e:
            logger.error(f"Error reducing dimensions: {e}")
            return np.array([])
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """
        Save embeddings to numpy file
        
        Args:
            embeddings: Embedding array
            output_path: Path to save file
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, embeddings)
            logger.info(f"Embeddings saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        Load embeddings from numpy file
        
        Args:
            file_path: Path to numpy file
        
        Returns:
            Loaded embeddings array
        """
        try:
            embeddings = np.load(file_path)
            logger.info(f"Loaded embeddings from {file_path}: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return np.array([])
    
    def detect_coordinated_content(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.9,
        min_cluster_size: int = 3
    ) -> List[List[int]]:
        """
        Detect groups with highly similar content (potential coordination)
        
        Args:
            embeddings: Embedding matrix
            threshold: Similarity threshold
            min_cluster_size: Minimum coordinated group size
        
        Returns:
            List of coordinated groups (lists of indices)
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Compute similarity matrix
            sim_matrix = cosine_similarity(embeddings)
            
            # Find highly similar pairs
            coordinated_groups = []
            processed = set()
            
            for i in range(len(embeddings)):
                if i in processed:
                    continue
                
                # Find all indices similar to i
                similar_indices = np.where(sim_matrix[i] >= threshold)[0]
                similar_indices = [idx for idx in similar_indices if idx not in processed]
                
                if len(similar_indices) >= min_cluster_size:
                    coordinated_groups.append(similar_indices.tolist())
                    processed.update(similar_indices)
            
            logger.info(f"Detected {len(coordinated_groups)} coordinated groups")
            return coordinated_groups
            
        except Exception as e:
            logger.error(f"Error detecting coordinated content: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Test configuration
    test_config = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'batch_size': 16,
        'device': 'auto'
    }
    
    # Test texts
    test_texts = [
        "Artificial intelligence is transforming technology",
        "AI is changing the world of technology",
        "Machine learning is a subset of AI",
        "The weather is nice today",
        "It's a beautiful sunny day"
    ]
    
    embedder = BERTEmbedder(test_config)
    
    # Generate embeddings
    embeddings = embedder.encode_texts(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Compute similarity
    sim_matrix = embedder.compute_similarity_matrix(embeddings)
    print(f"\nSimilarity Matrix:\n{sim_matrix}")
    
    # Detect coordination
    coordinated = embedder.detect_coordinated_content(embeddings, threshold=0.7)
    print(f"\nCoordinated groups: {coordinated}")