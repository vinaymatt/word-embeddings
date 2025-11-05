"""
Protein Ranking Module

Ranks proteins based on similarity scores from intersection networks.
Filters results using Named Entity Recognition.
"""
import os
import pickle
import json
from typing import List, Dict, Tuple
from pathlib import Path
import argparse
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm
import spacy

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import ensure_directory


class ProteinRanker:
    """
    Rank proteins from similarity networks using NER filtering
    """
    
    def __init__(
        self,
        config: dict,
        logger=None
    ):
        """
        Initialize ranker
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)
        
        # Load NER model
        ner_config = config['ner']
        self.ner_model_name = ner_config['models'][0]
        
        self.logger.info(f"Loading NER model: {self.ner_model_name}")
        try:
            self.nlp = spacy.load(self.ner_model_name)
        except OSError:
            self.logger.warning(f"Model {self.ner_model_name} not found. Using en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
        
        self.entity_types = ner_config['entity_types']
        
        # Ranking parameters
        rank_config = config['ranking']
        self.top_n = rank_config['top_n_proteins']
        self.use_ner = rank_config['use_ner']
        
        self.logger.info(f"Initialized protein ranker (top_n={self.top_n})")
    
    def is_protein_or_gene(
        self,
        term: str
    ) -> bool:
        """
        Check if a term is a protein or gene using NER
        
        Args:
            term: Term to check
        
        Returns:
            True if term is protein/gene
        """
        # Clean up term (replace underscores with spaces)
        cleaned = term.replace('_', ' ')
        
        # Process with NER
        doc = self.nlp(cleaned)
        
        # Check entities
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                return True
        
        # Also check if term contains known protein patterns
        # (all caps, ends with numbers, etc.)
        if term.isupper() and len(term) > 2:
            return True
        
        return False
    
    def rank_from_similarity_scores(
        self,
        similarity_scores: Dict[str, float],
        filter_proteins: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rank terms by similarity score
        
        Args:
            similarity_scores: Dictionary of term -> score
            filter_proteins: If True, filter to proteins only
        
        Returns:
            List of (term, score) tuples, sorted by score
        """
        self.logger.info(f"Ranking {len(similarity_scores)} terms")
        
        # Filter to proteins if requested
        if filter_proteins and self.use_ner:
            self.logger.info("Filtering to proteins/genes using NER...")
            
            filtered_scores = {}
            for term, score in tqdm(similarity_scores.items(), desc="NER filtering"):
                if self.is_protein_or_gene(term):
                    filtered_scores[term] = score
            
            self.logger.info(f"  Found {len(filtered_scores)} proteins/genes")
            similarity_scores = filtered_scores
        
        # Sort by score
        ranked = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N
        top_ranked = ranked[:self.top_n]
        
        self.logger.info(f"Top {len(top_ranked)} proteins:")
        for i, (term, score) in enumerate(top_ranked[:10], 1):
            self.logger.info(f"  {i}. {term}: {score:.4f}")
        
        return top_ranked
    
    def rank_by_year(
        self,
        embeddings_dir: str,
        input_words: List[str],
        start_year: int,
        end_year: int,
        top_n_per_word: int = 50
    ) -> pd.DataFrame:
        """
        Rank proteins year by year
        
        Args:
            embeddings_dir: Directory with embedding files
            input_words: List of query words
            start_year: Starting year
            end_year: Ending year
            top_n_per_word: Top N similar words per input word
        
        Returns:
            DataFrame with year-by-year rankings
        """
        self.logger.info(f"Ranking proteins for years {start_year}-{end_year}")
        
        # Import here to avoid circular dependency
        from src.graph_construction.similarity_network import SimilarityNetworkBuilder
        
        network_config = self.config  # Use same config
        builder = SimilarityNetworkBuilder(network_config, self.logger)
        
        # Collect rankings by year
        rankings_by_year = {}
        
        for year in tqdm(range(start_year, end_year + 1), desc="Processing years"):
            # Find embedding file
            import glob
            pattern = os.path.join(embeddings_dir, f'embeddings_{year}_*.pkl')
            matching = glob.glob(pattern)
            
            if not matching:
                continue
            
            # Load embeddings
            embeddings = builder.load_embeddings(matching[0])
            
            # Build intersection network
            _, similarity_scores = builder.build_intersection_network(
                embeddings,
                input_words,
                top_n_per_word
            )
            
            # Rank proteins
            ranked = self.rank_from_similarity_scores(
                similarity_scores,
                filter_proteins=True
            )
            
            rankings_by_year[year] = ranked
        
        # Convert to DataFrame
        df = self._rankings_to_dataframe(rankings_by_year)
        
        return df
    
    def _rankings_to_dataframe(
        self,
        rankings_by_year: Dict[int, List[Tuple[str, float]]]
    ) -> pd.DataFrame:
        """
        Convert year-by-year rankings to DataFrame
        
        Args:
            rankings_by_year: Dictionary of year -> rankings
        
        Returns:
            DataFrame with columns: Year, Rank, Protein, Score
        """
        rows = []
        
        for year, rankings in rankings_by_year.items():
            for rank, (protein, score) in enumerate(rankings, 1):
                rows.append({
                    'Year': year,
                    'Rank': rank,
                    'Protein': protein,
                    'Score': score
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def save_rankings(
        self,
        rankings_df: pd.DataFrame,
        output_file: str
    ):
        """
        Save rankings to file
        
        Args:
            rankings_df: Rankings DataFrame
            output_file: Output file path (CSV or Excel)
        """
        ensure_directory(os.path.dirname(output_file))
        
        if output_file.endswith('.csv'):
            rankings_df.to_csv(output_file, index=False)
        elif output_file.endswith('.xlsx'):
            rankings_df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported file format: {output_file}")
        
        self.logger.info(f"Saved rankings to {output_file}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Rank proteins from similarity networks"
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
        help='Output file for rankings (CSV or Excel)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2000,
        help='Starting year'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2023,
        help='Ending year'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = setup_logger(
        'protein_ranking',
        log_file=os.path.join(os.path.dirname(args.output), 'protein_ranking.log')
    )
    
    # Create ranker
    ranker = ProteinRanker(config, logger)
    
    # Get input words from config (case study)
    input_words = config['case_study_eNOS']['input_words']
    logger.info(f"Using {len(input_words)} input words: {', '.join(input_words[:3])}...")
    
    # Rank proteins year by year
    rankings_df = ranker.rank_by_year(
        args.embeddings,
        input_words,
        args.start_year,
        args.end_year
    )
    
    # Save rankings
    ranker.save_rankings(rankings_df, args.output)
    
    logger.info("Protein ranking complete!")


if __name__ == '__main__':
    main()

