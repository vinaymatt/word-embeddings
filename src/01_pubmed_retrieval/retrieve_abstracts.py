"""
PubMed Abstract Retrieval Module

Retrieves abstracts from PubMed using the Entrez API with year/month/day batching
for large result sets.
"""
import os
import time
from calendar import monthrange
from typing import Optional
from pathlib import Path
import argparse
import yaml

from Bio import Entrez
import pubmed_parser as pp

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import ensure_directory


class PubMedRetriever:
    """
    Retrieve abstracts from PubMed with intelligent batching for large result sets
    """
    
    def __init__(
        self,
        email: str,
        search_query: str,
        output_dir: str,
        batch_size: int = 10000,
        delay: int = 5,
        logger=None
    ):
        """
        Initialize PubMed retriever
        
        Args:
            email: Email for Entrez API (required by NCBI)
            search_query: PubMed search query string
            output_dir: Directory to save abstracts
            batch_size: Maximum results per API call
            delay: Seconds to wait between API calls
            logger: Optional logger instance
        """
        self.email = email
        self.search_query = search_query
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.delay = delay
        self.logger = logger or setup_logger(__name__)
        
        # Set Entrez email
        Entrez.email = self.email
        
        # Create output directory
        ensure_directory(self.output_dir)
    
    def retrieve_pmids_for_year(
        self,
        year: int
    ) -> list:
        """
        Retrieve PubMed IDs for a given year
        
        Args:
            year: Year to retrieve
        
        Returns:
            List of PMIDs
        """
        self.logger.info(f"Fetching PMIDs for year {year}")
        
        # Try year-level search first
        handle = Entrez.esearch(
            db="pubmed",
            term=f"{self.search_query} AND {year}[PDAT]",
            retmax=self.batch_size
        )
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record["IdList"]
        count = len(pmids)
        
        self.logger.info(f"Number of search results for {year}: {count}")
        
        # If results are below threshold, return them
        if count < 9999:
            return pmids
        
        # Otherwise, batch by month
        all_pmids = []
        for month in range(1, 13):
            month_pmids = self._retrieve_pmids_for_month(year, month)
            all_pmids.extend(month_pmids)
            
            if len(month_pmids) > 0:
                time.sleep(self.delay)
        
        return all_pmids
    
    def _retrieve_pmids_for_month(
        self,
        year: int,
        month: int
    ) -> list:
        """
        Retrieve PMIDs for a specific month
        
        Args:
            year: Year
            month: Month (1-12)
        
        Returns:
            List of PMIDs
        """
        handle = Entrez.esearch(
            db="pubmed",
            term=f"{self.search_query} AND {year}/{month:02d}[PDAT]",
            retmax=self.batch_size
        )
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record["IdList"]
        count = len(pmids)
        
        self.logger.info(f"Number of search results for {year}-{month:02d}: {count}")
        
        # If results below threshold, return
        if count < 9999:
            return pmids
        
        # Otherwise, batch by day
        all_pmids = []
        num_days = monthrange(year, month)[1]
        
        for day in range(1, num_days + 1):
            day_pmids = self._retrieve_pmids_for_day(year, month, day)
            all_pmids.extend(day_pmids)
            
            if len(day_pmids) == self.batch_size:
                time.sleep(self.delay)
        
        return all_pmids
    
    def _retrieve_pmids_for_day(
        self,
        year: int,
        month: int,
        day: int
    ) -> list:
        """
        Retrieve PMIDs for a specific day
        
        Args:
            year: Year
            month: Month (1-12)
            day: Day of month
        
        Returns:
            List of PMIDs
        """
        handle = Entrez.esearch(
            db="pubmed",
            term=f"{self.search_query} AND {year}/{month:02d}/{day:02d}[PDAT]",
            retmax=self.batch_size
        )
        record = Entrez.read(handle)
        handle.close()
        
        pmids = record["IdList"]
        
        self.logger.info(
            f"Number of search results for {year}-{month:02d}-{day:02d}: {len(pmids)}"
        )
        
        return pmids
    
    def fetch_abstracts_from_pmids(
        self,
        pmids: list,
        year: int
    ):
        """
        Fetch abstracts for a list of PMIDs and save to file
        
        Args:
            pmids: List of PubMed IDs
            year: Year (for output filename)
        """
        output_file = os.path.join(self.output_dir, f"{year}.txt")
        
        self.logger.info(f"Fetching {len(pmids)} abstracts for year {year}")
        
        # Fetch in batches
        for i in range(0, len(pmids), self.batch_size):
            pmids_batch = pmids[i:i + self.batch_size]
            
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=pmids_batch,
                    rettype="medline",
                    retmode="xml"
                )
                xml_data = handle.read()
                handle.close()
                
                # Parse MEDLINE XML
                articles = pp.parse_medline_xml(xml_data)
                
                # Save abstracts
                with open(output_file, 'a', encoding='utf-8') as f:
                    for article in articles:
                        abstract = article.get('abstract', '')
                        
                        if abstract:
                            # Clean newlines
                            abstract_clean = abstract.replace('\n', ' ')
                            f.write(abstract_clean)
                            f.write('\n\n')
                
                # Rate limiting
                if i + self.batch_size < len(pmids):
                    time.sleep(self.delay)
            
            except Exception as e:
                self.logger.error(f"Error fetching batch {i}-{i+self.batch_size}: {e}")
                continue
    
    def retrieve_years(
        self,
        start_year: int,
        end_year: int,
        fetch_abstracts: bool = True
    ):
        """
        Retrieve abstracts for a range of years
        
        Args:
            start_year: Starting year (inclusive)
            end_year: Ending year (exclusive)
            fetch_abstracts: If True, fetch full abstracts; if False, only PMIDs
        """
        for year in range(start_year, end_year):
            try:
                pmids = self.retrieve_pmids_for_year(year)
                
                if fetch_abstracts and len(pmids) > 0:
                    self.fetch_abstracts_from_pmids(pmids, year)
                
                self.logger.info(f"Completed year {year}: {len(pmids)} PMIDs")
            
            except Exception as e:
                self.logger.error(f"Error processing year {year}: {e}")
                continue


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Retrieve abstracts from PubMed"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for abstracts'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=1929,
        help='Starting year (default: 1929)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2024,
        help='Ending year (default: 2024)'
    )
    parser.add_argument(
        '--email',
        type=str,
        help='Email for Entrez API (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get parameters
    email = args.email or config.get('email', 'your.email@example.com')
    search_query = config.get('search_query', '')
    batch_size = config.get('batch_size', 10000)
    delay = config.get('delay', 5)
    
    # Create logger
    logger = setup_logger(
        'pubmed_retrieval',
        log_file=os.path.join(args.output, 'retrieval.log')
    )
    
    # Create retriever
    retriever = PubMedRetriever(
        email=email,
        search_query=search_query,
        output_dir=args.output,
        batch_size=batch_size,
        delay=delay,
        logger=logger
    )
    
    # Retrieve abstracts
    logger.info(f"Starting retrieval: {args.start_year}-{args.end_year}")
    logger.info(f"Search query: {search_query}")
    
    retriever.retrieve_years(args.start_year, args.end_year)
    
    logger.info("Retrieval complete!")


if __name__ == '__main__':
    main()

