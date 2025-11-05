"""
Skip-Gram Embedding Training Module

Trains year-wise cumulative Skip-Gram embeddings on preprocessed abstracts.
Each year's model is trained on all abstracts from start_year to that year.
"""
import os
import re
import pickle
import random
from typing import List, Dict
from pathlib import Path
import argparse
import yaml

import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.callbacks import CallbackAny2Vec

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import read_abstracts_by_year, ensure_directory


class TqdmWord2VecCallback(CallbackAny2Vec):
    """Callback for displaying training progress with tqdm"""
    
    def __init__(self, epochs: int, total_examples: int):
        self.epochs = epochs
        self.total_examples = total_examples
        self.pbar = tqdm(total=total_examples, desc="Training Word2Vec", unit="examples")
    
    def on_epoch_end(self, model):
        self.pbar.update(self.total_examples // self.epochs)
    
    def on_train_end(self, model):
        self.pbar.close()


class SkipGramTrainer:
    """
    Train Skip-Gram word embeddings with year-wise cumulative training
    """
    
    def __init__(
        self,
        config: dict,
        logger=None
    ):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)
        
        # Extract hyperparameters
        sg_config = config['skip_gram']
        self.embedding_size = sg_config['embedding_size']
        self.window_size = sg_config['window_size']
        self.min_count = sg_config['min_count']
        self.negative_sampling = sg_config['negative_sampling']
        self.epochs = sg_config['epochs']
        self.learning_rate_initial = sg_config['learning_rate_initial']
        self.learning_rate_final = sg_config['learning_rate_final']
        self.fixed_lr = sg_config['fixed_learning_rate']
        self.decay_rate = sg_config['decay_rate']
        self.workers = sg_config['workers']
        self.seed = sg_config['seed']
        
        # Phrase detection
        phrase_config = config['phrases']
        self.detect_phrases = phrase_config['detect_phrases']
        self.phrase_min_count = phrase_config['min_count']
        self.phrase_threshold = phrase_config['threshold']
        
        # Temporal settings
        temporal_config = config['temporal']
        self.cumulative = temporal_config['cumulative']
        self.start_year = temporal_config['start_year']
        self.end_year = temporal_config['end_year']
        
        # Tokenization
        token_config = config['tokenization']
        self.token_pattern = token_config['pattern']
        
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.logger.info(f"Initialized Skip-Gram trainer with embedding_size={self.embedding_size}, "
                        f"window={self.window_size}, seed={self.seed}")
    
    def tokenize_abstracts(
        self,
        abstracts: List[str]
    ) -> List[List[str]]:
        """
        Tokenize abstracts using regex pattern
        
        Args:
            abstracts: List of abstract strings
        
        Returns:
            List of tokenized abstracts (list of token lists)
        """
        self.logger.info(f"Tokenizing {len(abstracts)} abstracts...")
        
        tokenized = [
            re.findall(self.token_pattern, abstract)
            for abstract in tqdm(abstracts, desc="Tokenizing")
        ]
        
        return tokenized
    
    def detect_phrases(
        self,
        tokenized_abstracts: List[List[str]]
    ) -> Phraser:
        """
        Detect bigrams and trigrams
        
        Args:
            tokenized_abstracts: List of tokenized abstracts
        
        Returns:
            Phraser object for applying phrases
        """
        self.logger.info(
            f"Detecting phrases (min_count={self.phrase_min_count}, "
            f"threshold={self.phrase_threshold})"
        )
        
        phrases = Phrases(
            tokenized_abstracts,
            min_count=self.phrase_min_count,
            threshold=self.phrase_threshold
        )
        phraser = Phraser(phrases)
        
        self.logger.info(f"Detected {len(phraser.phrasegrams)} phrase patterns")
        
        return phraser
    
    def apply_phrases(
        self,
        tokenized_abstracts: List[List[str]],
        phraser: Phraser
    ) -> List[List[str]]:
        """
        Apply phrase detection to tokenized abstracts
        
        Args:
            tokenized_abstracts: List of tokenized abstracts
            phraser: Phraser object
        
        Returns:
            List of tokenized abstracts with phrases merged
        """
        self.logger.info("Applying phrase detection...")
        
        return [phraser[abstract] for abstract in tqdm(tokenized_abstracts, desc="Applying phrases")]
    
    def train_skipgram(
        self,
        sentences: List[List[str]],
        year: int
    ) -> Word2Vec:
        """
        Train Skip-Gram model
        
        Args:
            sentences: List of tokenized sentences
            year: Year being trained (for logging)
        
        Returns:
            Trained Word2Vec model
        """
        self.logger.info(f"Training Skip-Gram for year {year}")
        self.logger.info(f"  Corpus size: {len(sentences)} documents")
        
        # Create callback for progress tracking
        total_examples = len(sentences)
        
        # Train model
        model = Word2Vec(
            sentences=sentences,
            sg=1,  # Skip-Gram (not CBOW)
            vector_size=self.embedding_size,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.fixed_lr,  # Use fixed learning rate
            min_alpha=self.learning_rate_final,
            sample=self.decay_rate,
            negative=self.negative_sampling,
            seed=self.seed,
            callbacks=[TqdmWord2VecCallback(self.epochs, total_examples)]
        )
        
        vocab_size = len(model.wv)
        self.logger.info(f"  Vocabulary size: {vocab_size}")
        
        return model
    
    def save_embeddings(
        self,
        model: Word2Vec,
        year: int,
        output_dir: str
    ):
        """
        Save trained embeddings
        
        Args:
            model: Trained Word2Vec model
            year: Year
            output_dir: Output directory
        """
        ensure_directory(output_dir)
        
        # Create filename with hyperparameters
        filename = (
            f"embeddings_{year}_"
            f"lr{self.fixed_lr}_"
            f"dr{self.decay_rate}_"
            f"ed{self.embedding_size}_"
            f"ns{self.negative_sampling}.pkl"
        )
        
        output_path = os.path.join(output_dir, filename)
        
        # Save word vectors only (not full model)
        with open(output_path, 'wb') as f:
            pickle.dump(model.wv, f)
        
        self.logger.info(f"Saved embeddings to {output_path}")
    
    def train_cumulative(
        self,
        abstracts_by_year: Dict[int, List[str]],
        output_dir: str
    ):
        """
        Train embeddings cumulatively by year
        
        Args:
            abstracts_by_year: Dictionary mapping year to abstracts
            output_dir: Output directory for embeddings
        """
        self.logger.info("Starting cumulative training")
        self.logger.info(f"Year range: {self.start_year}-{self.end_year}")
        
        for year in range(self.start_year, self.end_year + 1):
            # Skip if no data for this year
            if year not in abstracts_by_year:
                self.logger.warning(f"No abstracts for year {year}, skipping")
                continue
            
            # Collect cumulative abstracts
            cumulative_abstracts = []
            for y in sorted(abstracts_by_year.keys()):
                if y <= year:
                    cumulative_abstracts.extend(abstracts_by_year[y])
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Year {year}: {len(cumulative_abstracts)} cumulative abstracts")
            self.logger.info(f"{'='*60}")
            
            # Tokenize
            tokenized = self.tokenize_abstracts(cumulative_abstracts)
            
            # Detect and apply phrases
            if self.detect_phrases:
                phraser = self.detect_phrases(tokenized)
                tokenized_with_phrases = self.apply_phrases(tokenized, phraser)
            else:
                tokenized_with_phrases = tokenized
            
            # Train model
            model = self.train_skipgram(tokenized_with_phrases, year)
            
            # Save embeddings
            self.save_embeddings(model, year, output_dir)
            
            self.logger.info(f"Completed year {year}\n")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Train Skip-Gram word embeddings"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to embedding configuration YAML'
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
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Starting year (overrides config)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='Ending year (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line args
    if args.start_year:
        config['temporal']['start_year'] = args.start_year
    if args.end_year:
        config['temporal']['end_year'] = args.end_year
    if args.seed:
        config['skip_gram']['seed'] = args.seed
    
    # Create logger
    logger = setup_logger(
        'skipgram_training',
        log_file=os.path.join(args.output, 'training.log')
    )
    
    # Create trainer
    trainer = SkipGramTrainer(config, logger)
    
    # Load abstracts
    logger.info("Loading abstracts from directory...")
    abstracts_by_year = read_abstracts_by_year(args.input)
    logger.info(f"Loaded {len(abstracts_by_year)} years")
    
    total_abstracts = sum(len(abstracts) for abstracts in abstracts_by_year.values())
    logger.info(f"Total abstracts: {total_abstracts:,}")
    
    # Train
    trainer.train_cumulative(abstracts_by_year, args.output)
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()

