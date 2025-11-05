"""
File I/O utilities for reading and writing abstracts
"""
import os
from pathlib import Path
from typing import List, Dict


def read_abstracts_from_file(file_path: str) -> List[str]:
    """
    Read abstracts from a text file where abstracts are separated by double newlines
    
    Args:
        file_path: Path to the text file
    
    Returns:
        List of abstract strings
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines
    abstracts = content.split('\n\n')
    
    # Filter out empty strings
    abstracts = [abstract.strip() for abstract in abstracts if abstract.strip()]
    
    return abstracts


def write_abstracts_to_file(abstracts: List[str], file_path: str, append: bool = False):
    """
    Write abstracts to a text file, separated by double newlines
    
    Args:
        abstracts: List of abstract strings
        file_path: Output file path
        append: If True, append to existing file; otherwise overwrite
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    mode = 'a' if append else 'w'
    
    with open(file_path, mode, encoding='utf-8') as f:
        for abstract in abstracts:
            f.write(abstract)
            f.write('\n\n')


def read_abstracts_by_year(directory: str, year_pattern: str = "{year}.txt") -> Dict[int, List[str]]:
    """
    Read abstracts organized by year from a directory
    
    Args:
        directory: Directory containing year-based text files
        year_pattern: Pattern for year files (e.g., "{year}.txt" or "abstracts_{year}.txt")
    
    Returns:
        Dictionary mapping year (int) to list of abstracts
    """
    abstracts_by_year = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Try to extract year from filename
            try:
                # Handle various patterns
                if filename.startswith('english_'):
                    year_str = filename.replace('english_', '').replace('.txt', '')
                elif 'abstracts_' in filename:
                    year_str = filename.split('_')[1].replace('.txt', '')
                else:
                    year_str = filename.replace('.txt', '')
                
                year = int(year_str)
                
                file_path = os.path.join(directory, filename)
                abstracts_by_year[year] = read_abstracts_from_file(file_path)
            
            except (ValueError, IndexError):
                # Skip files that don't match expected pattern
                continue
    
    return abstracts_by_year


def ensure_directory(directory: str):
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

