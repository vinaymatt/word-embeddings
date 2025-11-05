"""
Apply Trained Classifier to Corpus

Applies the trained MNB classifier to filter the full preprocessed corpus.
"""
import os
import pickle
from typing import List
from pathlib import Path
import argparse
import yaml

import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import read_abstracts_by_year, write_abstracts_to_file, ensure_directory
from train_mnb import MNBClassifierTrainer


class ClassifierApplicator:
    """
    Apply trained classifier to filter corpus
    """
    
    def __init__(
        self,
        model_dir: str,
        config: dict,
        logger=None
    ):
        """
        Initialize applicator
        
        Args:
            model_dir: Directory with trained model
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.model_dir = model_dir
        self.config = config
        self.logger = logger or setup_logger(__name__)
        
        # Load classifier
        self.logger.info(f"Loading classifier from {model_dir}")
        self.trainer = MNBClassifierTrainer(config, logger)
        self.trainer.load_model(model_dir)
        
        # Extract settings
        prod_config = config['production']
        self.batch_size = prod_config['batch_size']
        self.save_predictions = prod_config['save_predictions']
        self.save_probabilities = prod_config['save_probabilities']
    
    def classify_abstracts(
        self,
        abstracts: List[str],
        batch_size: int = None
    ) -> tuple:
        """
        Classify abstracts
        
        Args:
            abstracts: List of abstract texts
            batch_size: Batch size for processing
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        batch_size = batch_size or self.batch_size
        
        self.logger.info(f"Classifying {len(abstracts)} abstracts in batches of {batch_size}")
        
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in tqdm(range(0, len(abstracts), batch_size), desc="Classifying"):
            batch = abstracts[i:i+batch_size]
            
            # Predict
            preds, probs = self.trainer.predict(batch, return_proba=True)
            
            all_predictions.extend(preds)
            all_probabilities.extend(probs)
        
        return all_predictions, all_probabilities
    
    def filter_corpus_by_year(
        self,
        input_dir: str,
        output_dir: str,
        min_probability: float = 0.5
    ):
        """
        Filter preprocessed corpus by year
        
        Args:
            input_dir: Input directory with preprocessed abstracts
            output_dir: Output directory for classified abstracts
            min_probability: Minimum probability threshold for relevance
        """
        self.logger.info("Filtering corpus by year...")
        
        # Read abstracts by year
        abstracts_by_year = read_abstracts_by_year(input_dir)
        
        ensure_directory(output_dir)
        
        total_abstracts = 0
        total_retained = 0
        
        # Process each year
        for year in sorted(abstracts_by_year.keys()):
            abstracts = abstracts_by_year[year]
            
            self.logger.info(f"Year {year}: {len(abstracts)} abstracts")
            
            # Classify
            predictions, probabilities = self.classify_abstracts(abstracts)
            
            # Filter relevant abstracts
            relevant_abstracts = [
                abstract for abstract, pred, prob in zip(abstracts, predictions, probabilities)
                if pred == 1 and prob >= min_probability
            ]
            
            # Save filtered abstracts
            if relevant_abstracts:
                output_file = os.path.join(output_dir, f"{year}_abstracts_classified.txt")
                write_abstracts_to_file(relevant_abstracts, output_file)
            
            retention_rate = len(relevant_abstracts) / len(abstracts) * 100 if abstracts else 0
            
            self.logger.info(f"  Retained: {len(relevant_abstracts)}/{len(abstracts)} ({retention_rate:.1f}%)")
            
            total_abstracts += len(abstracts)
            total_retained += len(relevant_abstracts)
        
        overall_retention = total_retained / total_abstracts * 100 if total_abstracts else 0
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Overall: {total_retained:,}/{total_abstracts:,} retained ({overall_retention:.1f}%)")
        self.logger.info(f"{'='*60}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Apply trained classifier to filter corpus"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to classifier configuration YAML'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Directory with trained model'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with preprocessed abstracts'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for classified abstracts'
    )
    parser.add_argument(
        '--min-prob',
        type=float,
        default=0.5,
        help='Minimum probability threshold for relevance'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = setup_logger(
        'classifier_application',
        log_file=os.path.join(args.output, 'classification.log')
    )
    
    # Create applicator
    applicator = ClassifierApplicator(args.model, config, logger)
    
    # Filter corpus
    applicator.filter_corpus_by_year(
        args.input,
        args.output,
        args.min_prob
    )
    
    logger.info("Classification complete!")


if __name__ == '__main__':
    main()

