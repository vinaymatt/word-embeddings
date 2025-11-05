"""
Language detection and filtering
Removes non-English abstracts from the corpus
"""
import os
import re
from typing import List
from pathlib import Path
import argparse

from langdetect import detect

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import read_abstracts_from_file, write_abstracts_to_file, ensure_directory


def detect_language(text: str) -> str:
    """
    Detect language of text
    
    Args:
        text: Text to analyze
    
    Returns:
        Language code (e.g., 'en' for English)
    """
    try:
        return detect(text)
    except:
        return 'unknown'


def filter_english_abstracts(abstracts: List[str]) -> List[str]:
    """
    Filter list of abstracts to keep only English ones
    
    Args:
        abstracts: List of abstract strings
    
    Returns:
        List of English abstracts only
    """
    english_abstracts = []
    
    for abstract in abstracts:
        if abstract.strip() and detect_language(abstract) == 'en':
            english_abstracts.append(abstract)
    
    return english_abstracts


def process_directory(
    input_dir: str,
    output_dir: str,
    logger=None
):
    """
    Process all text files in a directory to filter English abstracts
    
    Args:
        input_dir: Input directory with abstract files
        output_dir: Output directory for filtered abstracts
        logger: Optional logger instance
    """
    logger = logger or setup_logger(__name__)
    
    ensure_directory(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            logger.info(f"Processing {filename}")
            
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"english_{filename}")
            
            # Read abstracts
            abstracts = read_abstracts_from_file(input_path)
            
            logger.info(f"  Total abstracts: {len(abstracts)}")
            
            # Filter English
            english_abstracts = filter_english_abstracts(abstracts)
            
            logger.info(f"  English abstracts: {len(english_abstracts)}")
            
            # Write output
            if english_abstracts:
                write_abstracts_to_file(english_abstracts, output_path)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Filter English abstracts from PubMed data"
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
        help='Output directory for English abstracts'
    )
    
    args = parser.parse_args()
    
    logger = setup_logger(
        'language_filter',
        log_file=os.path.join(args.output, 'language_filter.log')
    )
    
    logger.info("Starting English language filtering")
    process_directory(args.input, args.output, logger)
    logger.info("Filtering complete!")


if __name__ == '__main__':
    main()

