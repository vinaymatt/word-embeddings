"""
Main preprocessing module

Handles:
- Metadata tag removal (BACKGROUND/AIMS, RESULTS, etc.)
- Lemmatization (Spacy)
- Number tokenization (<num>)
- Punctuation removal
- Bigram/trigram detection (Gensim)
- Stopword filtering
- Selective lowercasing + NER
"""
import os
import re
from typing import List, Dict
from pathlib import Path
import argparse
import yaml
from collections import defaultdict

import spacy
from gensim.models.phrases import Phrases, Phraser
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import (
    read_abstracts_by_year,
    write_abstracts_to_file,
    ensure_directory
)


class AbstractPreprocessor:
    """
    Preprocess biomedical abstracts for embedding training
    """
    
    def __init__(
        self,
        config: dict,
        logger=None
    ):
        """
        Initialize preprocessor
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)
        
        # Load Spacy models for NER
        self.logger.info("Loading Spacy NER models...")
        try:
            self.nlp_jnlpba = spacy.load('en_ner_jnlpba_md')
            self.nlp_bionlp = spacy.load('en_ner_bionlp13cg_md')
        except OSError:
            self.logger.warning("Biomedical NER models not found. Using en_core_web_sm only.")
            self.nlp_jnlpba = None
            self.nlp_bionlp = None
        
        # Load general model
        try:
            self.nlp_general = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.error("en_core_web_sm not found. Please install: python -m spacy download en_core_web_sm")
            raise
        
        # Entity set for preserving casing
        self.entities = set()
        
        # Metadata tags to remove
        self.unwanted_tags = [
            'BACKGROUND/AIMS', 'RESULTS', 'METHODS', 'CONCLUSIONS',
            'METHODOLOGY/PRINCIPAL', 'FINDINGS', 'CONCLUSIONS/SIGNIFICANCE',
            'DESIGN/METHODS', 'DESIGN', 'PRINCIPAL', 'SIGNIFICANCE',
            'AIMS', 'CONCLUSION/SIGNIFICANCE', 'MATERIALS', 'MATERIAL',
            'OBJECTIVE', 'PURPOSE', 'INTRODUCTION', 'AIM', 'BACKGROUND',
            'CONCLUSION', 'METHOD', 'STUDY'
        ]
    
    def remove_metadata_tags(self, text: str) -> str:
        """
        Remove structural metadata tags from abstracts
        
        Args:
            text: Abstract text
        
        Returns:
            Cleaned text
        """
        words = text.split()
        words = [word for word in words if word not in self.unwanted_tags]
        return ' '.join(words)
    
    def clean_punctuation_and_numbers(self, text: str) -> str:
        """
        Clean punctuation and replace standalone numbers
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text with <num> tokens
        """
        # Replace standalone numbers with <num> token
        text = re.sub(r'(?<=\s)\d*\.?\d+%?(?=\s)', '<num>', text)
        
        # Remove commas
        text = re.sub(r',', '', text)
        
        # Remove periods at end of words (but keep decimals)
        text = re.sub(r'\.(?=\s|$)', '', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def is_mixed_case(self, word: str) -> bool:
        """Check if word has mixed case (likely an entity)"""
        if len(word) == 0:
            return False
        uppercase_count = sum(1 for letter in word if letter.isupper())
        ratio = uppercase_count / len(word)
        return 0.5 < ratio < 0.99
    
    def is_abbreviation(self, word: str) -> bool:
        """Check if word is an abbreviation (preserve casing)"""
        return word.isupper() or (len(word) == 1 and word.isalpha() and word.isupper()) or self.is_mixed_case(word)
    
    def extract_entities(self, text: str):
        """
        Extract biomedical entities using NER models
        
        Args:
            text: Input text
        """
        # Use biomedical NER models if available
        if self.nlp_jnlpba:
            doc = self.nlp_jnlpba(text)
            for ent in doc.ents:
                if ent.label_ in ['PROTEIN', 'GENE', 'DNA', 'RNA', 'CELL_LINE', 'CELL_TYPE']:
                    self.entities.add(ent.text)
        
        if self.nlp_bionlp:
            doc = self.nlp_bionlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PROTEIN', 'GENE']:
                    self.entities.add(ent.text)
        
        # Add abbreviations
        words = text.split()
        for word in words:
            if self.is_abbreviation(word):
                self.entities.add(word)
    
    def selectively_lowercase(self, text: str) -> List[str]:
        """
        Lowercase text while preserving entities and abbreviations
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Extract entities first
        self.extract_entities(text)
        
        # Tokenize
        words = text.split()
        
        # Lowercase non-entities
        processed_words = []
        for word in words:
            if word in self.entities or self.is_abbreviation(word):
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        
        return processed_words
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens while preserving entities
        
        Args:
            tokens: List of tokens
        
        Returns:
            Lemmatized tokens
        """
        # Create doc from tokens
        doc = self.nlp_general(' '.join(tokens))
        
        lemmatized = []
        for token in doc:
            # Preserve entities
            if token.text in self.entities:
                lemmatized.append(token.text)
            else:
                # Lemmatize
                lemmatized.append(token.lemma_)
        
        return lemmatized
    
    def preprocess_abstract(self, abstract: str) -> str:
        """
        Apply full preprocessing pipeline to an abstract
        
        Args:
            abstract: Raw abstract text
        
        Returns:
            Preprocessed abstract
        """
        # Remove metadata tags
        text = self.remove_metadata_tags(abstract)
        
        # Clean punctuation and numbers
        text = self.clean_punctuation_and_numbers(text)
        
        # Selective lowercasing
        tokens = self.selectively_lowercase(text)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back
        return ' '.join(tokens)
    
    def process_abstracts_by_year(
        self,
        abstracts_by_year: Dict[int, List[str]]
    ) -> Dict[int, List[str]]:
        """
        Process all abstracts organized by year
        
        Args:
            abstracts_by_year: Dictionary mapping year to abstracts
        
        Returns:
            Dictionary mapping year to preprocessed abstracts
        """
        processed_by_year = defaultdict(list)
        
        for year in sorted(abstracts_by_year.keys()):
            abstracts = abstracts_by_year[year]
            self.logger.info(f"Processing year {year}: {len(abstracts)} abstracts")
            
            for abstract in tqdm(abstracts, desc=f"Year {year}"):
                try:
                    processed = self.preprocess_abstract(abstract)
                    if processed.strip():
                        processed_by_year[year].append(processed)
                except Exception as e:
                    self.logger.warning(f"Error processing abstract: {e}")
                    continue
        
        return dict(processed_by_year)
    
    def detect_and_merge_phrases(
        self,
        tokenized_abstracts: List[List[str]],
        min_count: int = 7,
        threshold: int = 15
    ) -> Phraser:
        """
        Detect bigrams and trigrams using Gensim Phrases
        
        Args:
            tokenized_abstracts: List of tokenized abstracts
            min_count: Minimum count for phrase
            threshold: Threshold for phrase detection
        
        Returns:
            Phraser model for applying phrases
        """
        self.logger.info(f"Detecting phrases (min_count={min_count}, threshold={threshold})")
        
        phrases = Phrases(tokenized_abstracts, min_count=min_count, threshold=threshold)
        phraser = Phraser(phrases)
        
        self.logger.info(f"Detected {len(phraser.phrasegrams)} phrases")
        
        return phraser


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Preprocess PubMed abstracts"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to preprocessing configuration YAML'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory with abstract files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for preprocessed abstracts'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = setup_logger(
        'preprocessing',
        log_file=os.path.join(args.output, 'preprocessing.log')
    )
    
    # Create preprocessor
    preprocessor = AbstractPreprocessor(config, logger)
    
    # Read abstracts
    logger.info("Reading abstracts from directory...")
    abstracts_by_year = read_abstracts_by_year(args.input)
    logger.info(f"Loaded {len(abstracts_by_year)} years")
    
    # Process
    logger.info("Preprocessing abstracts...")
    processed_by_year = preprocessor.process_abstracts_by_year(abstracts_by_year)
    
    # Save
    ensure_directory(args.output)
    for year, abstracts in processed_by_year.items():
        output_file = os.path.join(args.output, f"preprocessed_{year}.txt")
        write_abstracts_to_file(abstracts, output_file)
        logger.info(f"Saved {len(abstracts)} abstracts for year {year}")
    
    logger.info("Preprocessing complete!")


if __name__ == '__main__':
    main()

