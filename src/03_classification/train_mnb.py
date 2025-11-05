"""
Multinomial Naive Bayes Classifier Training

Trains a simple, interpretable classifier for filtering relevant biomedical abstracts.
Uses bag-of-words features and provides transparent term-level importance.
"""
import os
import pickle
import random
from typing import List, Tuple
from pathlib import Path
import argparse
import yaml

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    cohen_kappa_score
)

import sys
sys.path.append(str(Path(__file__).parents[2]))
from utils.logging_setup import setup_logger
from utils.file_io import ensure_directory


class MNBClassifierTrainer:
    """
    Train Multinomial Naive Bayes classifier for abstract relevance
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
        
        # Extract parameters
        vec_config = config['vectorizer']
        self.max_features = vec_config['max_features']
        self.ngram_range = tuple(vec_config['ngram_range'])
        self.min_df = vec_config['min_df']
        
        # Training parameters
        train_config = config['training']
        self.seed = train_config['random_seed']
        
        # Set random seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize vectorizer and classifier
        self.vectorizer = None
        self.classifier = None
        
        self.logger.info(f"Initialized MNB trainer (seed={self.seed})")
    
    def load_labeled_data(
        self,
        data_file: str
    ) -> Tuple[List[str], List[int]]:
        """
        Load labeled training data
        
        Args:
            data_file: Path to labeled data (CSV or Excel)
        
        Returns:
            Tuple of (abstracts, labels)
        """
        self.logger.info(f"Loading labeled data from {data_file}")
        
        # Read file based on extension
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file}")
        
        # Expect columns: 'Abstract' and 'Relevance' (or 'Label')
        if 'Abstract' not in df.columns:
            raise ValueError("Data file must have 'Abstract' column")
        
        label_col = 'Relevance' if 'Relevance' in df.columns else 'Label'
        if label_col not in df.columns:
            raise ValueError(f"Data file must have '{label_col}' column")
        
        abstracts = df['Abstract'].tolist()
        labels = df[label_col].tolist()
        
        # Convert labels to binary if needed
        if not all(isinstance(l, int) for l in labels):
            # Map relevant/not_relevant to 1/0
            label_map = {'relevant': 1, 'not_relevant': 0, 'yes': 1, 'no': 0}
            labels = [label_map.get(str(l).lower(), int(l)) for l in labels]
        
        self.logger.info(f"Loaded {len(abstracts)} labeled abstracts")
        self.logger.info(f"  Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")
        
        return abstracts, labels
    
    def train(
        self,
        abstracts: List[str],
        labels: List[int],
        test_size: float = 0.2
    ):
        """
        Train classifier
        
        Args:
            abstracts: List of abstract texts
            labels: List of labels (0 or 1)
            test_size: Fraction of data for testing
        """
        self.logger.info("Training classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            abstracts, labels,
            test_size=test_size,
            random_state=self.seed,
            stratify=labels
        )
        
        self.logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Vectorize
        self.logger.info("Vectorizing text with CountVectorizer...")
        self.vectorizer = CountVectorizer(
            analyzer='word',
            token_pattern=r'\w{1,}',
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            binary=False
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        vocab_size = len(self.vectorizer.get_feature_names_out())
        self.logger.info(f"Vocabulary size: {vocab_size}")
        
        # Train classifier
        self.logger.info("Training Multinomial Naive Bayes...")
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        self.logger.info("\nEvaluation on test set:")
        y_pred = self.classifier.predict(X_test_vec)
        y_prob = self.classifier.predict_proba(X_test_vec)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        self.logger.info(f"  Accuracy:  {accuracy:.4f}")
        self.logger.info(f"  Precision: {precision:.4f}")
        self.logger.info(f"  Recall:    {recall:.4f}")
        self.logger.info(f"  F1-score:  {f1:.4f}")
        self.logger.info(f"  AUC-ROC:   {auc:.4f}")
        self.logger.info(f"  Cohen's Kappa: {kappa:.4f} ({kappa*100:.2f}%)")
        
        # Classification report
        self.logger.info("\nClassification Report:")
        report = classification_report(y_test, y_pred, target_names=['Not Relevant', 'Relevant'])
        self.logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.logger.info(f"\nConfusion Matrix:")
        self.logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        self.logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cohen_kappa': kappa
        }
    
    def get_feature_importance(
        self,
        top_n: int = 50
    ) -> Tuple[List[str], List[str]]:
        """
        Get most important features for each class
        
        Args:
            top_n: Number of top features per class
        
        Returns:
            Tuple of (top_positive_features, top_negative_features)
        """
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Classifier must be trained first")
        
        feature_names = self.vectorizer.get_feature_names_out()
        log_prob = self.classifier.feature_log_prob_
        
        # Class 0 (not relevant) features
        neg_indices = np.argsort(log_prob[0])[-top_n:][::-1]
        neg_features = [feature_names[i] for i in neg_indices]
        
        # Class 1 (relevant) features
        pos_indices = np.argsort(log_prob[1])[-top_n:][::-1]
        pos_features = [feature_names[i] for i in pos_indices]
        
        return pos_features, neg_features
    
    def save_model(
        self,
        output_dir: str,
        model_name: str = 'mnb_classifier'
    ):
        """
        Save trained model and vectorizer
        
        Args:
            output_dir: Output directory
            model_name: Base name for model files
        """
        ensure_directory(output_dir)
        
        # Save vectorizer
        vec_path = os.path.join(output_dir, f'{model_name}_vectorizer.pkl')
        with open(vec_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save classifier
        clf_path = os.path.join(output_dir, f'{model_name}_model.pkl')
        with open(clf_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        self.logger.info(f"Saved model to {output_dir}")
    
    def load_model(
        self,
        model_dir: str,
        model_name: str = 'mnb_classifier'
    ):
        """
        Load trained model and vectorizer
        
        Args:
            model_dir: Directory with model files
            model_name: Base name for model files
        """
        # Load vectorizer
        vec_path = os.path.join(model_dir, f'{model_name}_vectorizer.pkl')
        with open(vec_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load classifier
        clf_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        with open(clf_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        self.logger.info(f"Loaded model from {model_dir}")
    
    def predict(
        self,
        abstracts: List[str],
        return_proba: bool = False
    ):
        """
        Predict relevance for abstracts
        
        Args:
            abstracts: List of abstract texts
            return_proba: If True, return probabilities
        
        Returns:
            Predictions (and probabilities if requested)
        """
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Classifier must be trained or loaded first")
        
        # Vectorize
        X = self.vectorizer.transform(abstracts)
        
        # Predict
        predictions = self.classifier.predict(X)
        
        if return_proba:
            probabilities = self.classifier.predict_proba(X)[:, 1]
            return predictions, probabilities
        
        return predictions


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Train MNB classifier for abstract relevance"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to classifier configuration YAML'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to labeled data (CSV or Excel)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create logger
    logger = setup_logger(
        'mnb_training',
        log_file=os.path.join(args.output, 'training.log')
    )
    
    # Create trainer
    trainer = MNBClassifierTrainer(config, logger)
    
    # Load data
    abstracts, labels = trainer.load_labeled_data(args.data)
    
    # Train
    metrics = trainer.train(abstracts, labels, args.test_size)
    
    # Get feature importance
    logger.info("\nTop features for each class:")
    pos_features, neg_features = trainer.get_feature_importance(top_n=20)
    logger.info("  Relevant features: " + ", ".join(pos_features[:10]))
    logger.info("  Not relevant features: " + ", ".join(neg_features[:10]))
    
    # Save model
    trainer.save_model(args.output)
    
    logger.info("\nTraining complete!")


if __name__ == '__main__':
    main()

