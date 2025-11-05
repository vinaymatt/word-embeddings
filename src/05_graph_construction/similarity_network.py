"""
Similarity Network Construction

Builds weighted similarity networks from word embeddings using cosine similarity
and z-score normalization for cross-year comparison.
"""
import os
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import argparse
import yaml

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from scipy import stats
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import ensure_directory


class SimilarityNetworkBuilder:
    """
    Build weighted similarity networks from word embeddings
    """
    
    def __init__(
        self,
        config: dict,
        logger=None
    ):
        """
        Initialize network builder
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)
        
        # Extract parameters
        self.similarity_metric = config['similarity']['metric']
        self.normalize_vectors = config['similarity']['normalize']
        
        # Z-score settings
        self.use_z_score = config['z_score']['enabled']
        self.z_threshold = config['z_score']['threshold_std']
        
        # Network settings
        self.weighted = config['network']['weighted']
        self.directed = config['network']['directed']
        self.self_loops = config['network']['self_loops']
        
        # Layout settings
        self.layout_seed = config['layout']['seed']
        
        self.logger.info(f"Initialized network builder (z_threshold={self.z_threshold} std)")
    
    def load_embeddings(
        self,
        embedding_file: str
    ) -> Dict[str, np.ndarray]:
        """
        Load word embeddings from pickle file
        
        Args:
            embedding_file: Path to pickle file
        
        Returns:
            Dictionary mapping word to vector
        """
        self.logger.info(f"Loading embeddings from {embedding_file}")
        
        with open(embedding_file, 'rb') as f:
            word_vectors = pickle.load(f)
        
        # Convert to dictionary
        embeddings = {word: word_vectors[word] for word in word_vectors.key_to_index}
        
        self.logger.info(f"Loaded {len(embeddings)} word vectors")
        
        return embeddings
    
    def compute_similarity_matrix(
        self,
        embeddings: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute pairwise cosine similarity matrix
        
        Args:
            embeddings: Dictionary mapping word to vector
        
        Returns:
            Tuple of (similarity_matrix, word_list)
        """
        self.logger.info("Computing similarity matrix...")
        
        words = list(embeddings.keys())
        vectors = np.array([embeddings[word] for word in words])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(vectors)
        
        # Remove self-loops
        if not self.self_loops:
            np.fill_diagonal(similarity_matrix, 0)
        
        self.logger.info(f"Computed {len(words)}x{len(words)} similarity matrix")
        
        return similarity_matrix, words
    
    def z_score_normalization(
        self,
        similarity_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Z-score normalize similarity matrix
        
        Args:
            similarity_matrix: Raw similarity matrix
        
        Returns:
            Tuple of (z_scores, mean, std)
        """
        # Flatten matrix (exclude diagonal)
        n = similarity_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        similarities = similarity_matrix[mask]
        
        # Compute z-scores
        mean = np.mean(similarities)
        std = np.std(similarities)
        
        z_scores = (similarity_matrix - mean) / std
        
        self.logger.info(f"Z-score normalization: mean={mean:.4f}, std={std:.4f}")
        
        return z_scores, mean, std
    
    def build_network(
        self,
        similarity_matrix: np.ndarray,
        words: List[str],
        threshold: float = None
    ) -> nx.Graph:
        """
        Build network from similarity matrix
        
        Args:
            similarity_matrix: Similarity or z-score matrix
            words: List of words (node labels)
            threshold: Optional threshold for edge inclusion
        
        Returns:
            NetworkX graph
        """
        self.logger.info("Building network...")
        
        # Create graph
        if self.directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        for i, word in enumerate(words):
            G.add_node(i, label=word)
        
        # Add edges
        n = len(words)
        edge_count = 0
        
        for i in range(n):
            for j in range(i+1, n):  # Upper triangle only for undirected
                sim = similarity_matrix[i, j]
                
                # Apply threshold if specified
                if threshold is not None and sim < threshold:
                    continue
                
                # Add edge with weight
                if self.weighted:
                    G.add_edge(i, j, weight=float(sim))
                else:
                    G.add_edge(i, j)
                
                edge_count += 1
        
        self.logger.info(f"Created network: {len(words)} nodes, {edge_count} edges")
        
        return G
    
    def get_similar_words(
        self,
        embeddings: Dict[str, np.ndarray],
        query_word: str,
        top_n: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Get most similar words to a query word
        
        Args:
            embeddings: Word embeddings
            query_word: Query word
            top_n: Number of similar words to return
        
        Returns:
            List of (word, similarity) tuples
        """
        if query_word not in embeddings:
            self.logger.warning(f"Query word '{query_word}' not in vocabulary")
            return []
        
        query_vec = embeddings[query_word]
        
        # Compute similarities
        similarities = []
        for word, vec in embeddings.items():
            if word == query_word:
                continue
            
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            similarities.append((word, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_n]
    
    def build_intersection_network(
        self,
        embeddings: Dict[str, np.ndarray],
        input_words: List[str],
        top_n_per_word: int = 50,
        aggregation: str = 'mean'
    ) -> Tuple[nx.Graph, Dict[str, float]]:
        """
        Build intersection network from multiple input words
        
        Args:
            embeddings: Word embeddings
            input_words: List of seed words
            top_n_per_word: Top N similar words per input word
            aggregation: How to aggregate similarities ('mean', 'max', 'min')
        
        Returns:
            Tuple of (graph, similarity_scores)
        """
        self.logger.info(f"Building intersection network with {len(input_words)} input words")
        
        # Get similar words for each input word
        all_similar = {}
        for word in input_words:
            similar = self.get_similar_words(embeddings, word, top_n_per_word)
            all_similar[word] = dict(similar)
        
        # Aggregate similarities
        aggregated_scores = {}
        all_words = set()
        for similar_dict in all_similar.values():
            all_words.update(similar_dict.keys())
        
        for word in all_words:
            scores = []
            for input_word, similar_dict in all_similar.items():
                if word in similar_dict:
                    scores.append(similar_dict[word])
            
            if aggregation == 'mean':
                aggregated_scores[word] = np.mean(scores)
            elif aggregation == 'max':
                aggregated_scores[word] = np.max(scores)
            elif aggregation == 'min':
                aggregated_scores[word] = np.min(scores)
        
        # Build network with top words
        top_words = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Aggregated {len(aggregated_scores)} unique words")
        
        return top_words, aggregated_scores
    
    def save_network(
        self,
        G: nx.Graph,
        output_file: str,
        metadata: dict = None
    ):
        """
        Save network to JSON file
        
        Args:
            G: NetworkX graph
            output_file: Output file path
            metadata: Optional metadata to include
        """
        ensure_directory(os.path.dirname(output_file))
        
        # Convert to node-link format
        data = nx.node_link_data(G)
        
        # Add metadata
        if metadata:
            data['metadata'] = metadata
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Saved network to {output_file}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Build similarity networks from embeddings"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to network configuration YAML'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Directory with embedding files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for networks'
    )
    parser.add_argument(
        '--years',
        type=str,
        help='Comma-separated list of years (e.g., "2010,2015,2020")'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = setup_logger(
        'network_construction',
        log_file=os.path.join(args.output, 'network_construction.log')
    )
    
    # Create builder
    builder = SimilarityNetworkBuilder(config, logger)
    
    # Determine years to process
    if args.years:
        years = [int(y.strip()) for y in args.years.split(',')]
    else:
        # Process all embedding files
        import glob
        embedding_files = glob.glob(os.path.join(args.embeddings, 'embeddings_*_*.pkl'))
        years = [int(os.path.basename(f).split('_')[1]) for f in embedding_files]
        years = sorted(set(years))
    
    logger.info(f"Processing {len(years)} years")
    
    # Build networks for each year
    for year in tqdm(years, desc="Building networks"):
        # Find embedding file for this year
        embedding_pattern = os.path.join(args.embeddings, f'embeddings_{year}_*.pkl')
        import glob
        matching_files = glob.glob(embedding_pattern)
        
        if not matching_files:
            logger.warning(f"No embedding file found for year {year}")
            continue
        
        embedding_file = matching_files[0]
        
        # Load embeddings
        embeddings = builder.load_embeddings(embedding_file)
        
        # Compute similarity matrix
        sim_matrix, words = builder.compute_similarity_matrix(embeddings)
        
        # Z-score normalization
        if builder.use_z_score:
            z_scores, mean, std = builder.z_score_normalization(sim_matrix)
            threshold = builder.z_threshold  # Z-score threshold
            matrix_to_use = z_scores
        else:
            threshold = None
            matrix_to_use = sim_matrix
        
        # Build network
        G = builder.build_network(matrix_to_use, words, threshold)
        
        # Save network
        output_file = os.path.join(args.output, f'{year}_network.json')
        metadata = {
            'year': year,
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'z_score_enabled': builder.use_z_score
        }
        
        if builder.use_z_score:
            metadata['similarity_mean'] = float(mean)
            metadata['similarity_std'] = float(std)
        
        builder.save_network(G, output_file, metadata)
    
    logger.info("Network construction complete!")


if __name__ == '__main__':
    main()

