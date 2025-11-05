"""
GPT-4 Assisted Labeling Module

Uses OpenAI's GPT-4 to label abstracts as RELEVANT or NOT RELEVANT.
This creates the training dataset for the production MNB classifier.
"""
import os
import time
from typing import List, Dict
from pathlib import Path
import argparse
import yaml

import openai
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import ensure_directory


class GPT4Labeler:
    """
    Label abstracts using GPT-4 with retry logic and incremental saving
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0,
        max_retries: int = 5,
        logger=None
    ):
        """
        Initialize GPT-4 labeler
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")
            temperature: Temperature for generation (0 = deterministic)
            max_retries: Maximum retry attempts on errors
            logger: Optional logger instance
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.logger = logger or setup_logger(__name__)
        
        # Set API key
        openai.api_key = self.api_key
        
        self.logger.info(f"Initialized GPT-4 labeler (model={self.model}, temp={self.temperature})")
    
    def get_prompt(
        self,
        abstract: str,
        prompt_version: str = "refined"
    ) -> List[Dict[str, str]]:
        """
        Get prompt messages for GPT-4
        
        Args:
            abstract: Abstract text to classify
            prompt_version: Prompt version (initial, refined, few_shot)
        
        Returns:
            List of message dictionaries for ChatCompletion API
        """
        system_message = {"role": "system", "content": "You are a helpful assistant."}
        
        # Initial prompt (broad, had false positives)
        if prompt_version == "initial":
            user_content = (
                f"I am trying to train a skip gram model on a few abstracts, however this requires "
                f"that we don't use all abstracts but only a specific bunch based on relevance. "
                f"I want you to help me classify this abstract and respond to my query using a "
                f"RELEVANT or NOT RELEVANT answer (Mention these keywords). "
                f"Is this abstract relevant to the field of transmembrane proteins involved in the "
                f"transduction of fluid shear stress to nitric oxide production in endothelial cells? "
                f"This abstract is as follows: '{abstract}'."
            )
        
        # Refined prompt (changed "could relate" to "should relate")
        elif prompt_version == "refined":
            user_content = (
                f"As part of an academic research project that involves the study of transmembrane "
                f"proteins and their role in the transduction of fluid shear stress to nitric oxide "
                f"production in endothelial cells. The nature of my study necessitates the usage of "
                f"relevant literature, because we will be training a skip gram model from this corpus "
                f"of data. I would like you to help me in this selection process by classifying the "
                f"given abstract. Each abstract should be considered in terms of its potential value "
                f"to the model's training set, as we are applying a skip-gram model. "
                f"The criteria for relevance include: "
                f"Relating to the field of transmembrane proteins or "
                f"Relating to the transduction of fluid shear stress or "
                f"Relating to nitric oxide production in endothelial cells. "
                f"Respond only with RELEVANT OR NOT RELEVANT. "
                f"I do not want any explanation/justification of any sort in your answer. "
                f"Abstract: '{abstract}'."
            )
        
        # Few-shot prompt (with examples)
        elif prompt_version == "few_shot":
            # TODO: Add few-shot examples here
            user_content = user_content = (
                f"Classify the following abstract as RELEVANT or NOT RELEVANT based on "
                f"whether it relates to biomechanics, mechanobiology, transmembrane proteins, "
                f"fluid shear stress, or nitric oxide production in endothelial cells.\n\n"
                f"Respond ONLY with 'RELEVANT' or 'NOT RELEVANT'.\n\n"
                f"Abstract: '{abstract}'"
            )
        
        else:
            raise ValueError(f"Unknown prompt version: {prompt_version}")
        
        return [
            system_message,
            {"role": "user", "content": user_content}
        ]
    
    def label_abstract(
        self,
        abstract: str,
        prompt_version: str = "refined"
    ) -> str:
        """
        Label a single abstract using GPT-4
        
        Args:
            abstract: Abstract text
            prompt_version: Prompt version to use
        
        Returns:
            Label ('RELEVANT' or 'NOT RELEVANT')
        """
        messages = self.get_prompt(abstract, prompt_version)
        
        # Retry loop
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=10  # Short answer only
                )
                
                label = response['choices'][0]['message']['content'].strip().upper()
                
                # Normalize label
                if 'RELEVANT' in label and 'NOT' not in label:
                    return 'RELEVANT'
                elif 'NOT RELEVANT' in label or 'NOT_RELEVANT' in label:
                    return 'NOT RELEVANT'
                else:
                    self.logger.warning(f"Unexpected response: {label}")
                    return label
            
            except openai.error.ServiceUnavailableError as e:
                self.logger.warning(f"Attempt {attempt+1}/{self.max_retries}: Service unavailable - {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                else:
                    raise
            
            except Exception as e:
                self.logger.error(f"Error labeling abstract: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    raise
        
        return 'ERROR'
    
    def label_abstracts_batch(
        self,
        abstracts: List[str],
        output_file: str,
        prompt_version: str = "refined",
        resume: bool = True
    ):
        """
        Label a batch of abstracts and save incrementally
        
        Args:
            abstracts: List of abstract texts
            output_file: Output Excel file path
            prompt_version: Prompt version to use
            resume: If True, skip already labeled abstracts
        """
        self.logger.info(f"Labeling {len(abstracts)} abstracts")
        self.logger.info(f"Output: {output_file}")
        self.logger.info(f"Prompt version: {prompt_version}")
        
        # Load existing data if resuming
        if resume and os.path.isfile(output_file):
            self.logger.info("Resuming from existing file...")
            df = pd.read_excel(output_file)
        else:
            df = pd.DataFrame(columns=['Abstract', 'Relevance'])
        
        # Iterate over abstracts
        labeled_count = 0
        for abstract in tqdm(abstracts, desc="Labeling"):
            # Skip if already processed
            if df['Abstract'].isin([abstract]).any():
                continue
            
            # Label abstract
            try:
                label = self.label_abstract(abstract, prompt_version)
                
                # Append to dataframe
                df = pd.concat([
                    df,
                    pd.DataFrame([{'Abstract': abstract, 'Relevance': label}])
                ], ignore_index=True)
                
                # Save incrementally (prevent data loss)
                df.to_excel(output_file, index=False)
                
                labeled_count += 1
                
                # Rate limiting
                time.sleep(1)  # 1 second between requests
            
            except Exception as e:
                self.logger.error(f"Failed to label abstract: {e}")
                continue
        
        self.logger.info(f"Labeled {labeled_count} new abstracts")
        self.logger.info(f"Total in dataset: {len(df)}")
        
        return df
    
    def organize_labeled_data(
        self,
        labeled_file: str,
        output_dir: str
    ):
        """
        Organize labeled abstracts into relevant/not_relevant directories
        
        Args:
            labeled_file: Path to Excel file with labels
            output_dir: Base output directory
        """
        self.logger.info(f"Organizing labeled data from {labeled_file}")
        
        # Read labels
        df = pd.read_excel(labeled_file)
        
        # Create directories
        relevant_dir = os.path.join(output_dir, 'relevant')
        not_relevant_dir = os.path.join(output_dir, 'not_relevant')
        ensure_directory(relevant_dir)
        ensure_directory(not_relevant_dir)
        
        # Save abstracts to appropriate directories
        for i, row in df.iterrows():
            # Determine directory
            if row['Relevance'].strip().upper() == 'RELEVANT':
                dir_path = relevant_dir
            else:
                dir_path = not_relevant_dir
            
            # Save as text file
            file_path = os.path.join(dir_path, f'abstract{i+1}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(row['Abstract']))
        
        n_relevant = len(df[df['Relevance'].str.upper() == 'RELEVANT'])
        n_not_relevant = len(df[df['Relevance'].str.upper() != 'RELEVANT'])
        
        self.logger.info(f"Organized {n_relevant} relevant, {n_not_relevant} not relevant")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Label abstracts using GPT-4"
    )
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--abstracts',
        type=str,
        required=True,
        help='Path to text file with abstracts (separated by double newlines)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output Excel file for labeled data'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4',
        help='OpenAI model name (default: gpt-4)'
    )
    parser.add_argument(
        '--prompt-version',
        type=str,
        default='refined',
        choices=['initial', 'refined', 'few_shot'],
        help='Prompt version to use'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing output file'
    )
    parser.add_argument(
        '--organize',
        action='store_true',
        help='Organize labeled data into directories after labeling'
    )
    parser.add_argument(
        '--organize-output',
        type=str,
        help='Directory for organized data (required if --organize is set)'
    )
    
    args = parser.parse_args()
    
    # Create logger
    logger = setup_logger(
        'gpt4_labeling',
        log_file=os.path.join(os.path.dirname(args.output), 'gpt4_labeling.log')
    )
    
    # Create labeler
    labeler = GPT4Labeler(
        api_key=args.api_key,
        model=args.model,
        logger=logger
    )
    
    # Read abstracts
    logger.info(f"Reading abstracts from {args.abstracts}")
    with open(args.abstracts, 'r', encoding='utf-8') as f:
        content = f.read()
    
    abstracts = content.split('\n\n')
    abstracts = [a.strip() for a in abstracts if a.strip()]
    
    logger.info(f"Loaded {len(abstracts)} abstracts")
    
    # Label abstracts
    df = labeler.label_abstracts_batch(
        abstracts,
        args.output,
        args.prompt_version,
        args.resume
    )
    
    # Organize into directories if requested
    if args.organize:
        if not args.organize_output:
            raise ValueError("--organize-output required when --organize is set")
        
        labeler.organize_labeled_data(args.output, args.organize_output)
    
    logger.info("GPT-4 labeling complete!")


if __name__ == '__main__':
    main()

