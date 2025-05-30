# src/utils/data_loader.py
"""
Data loading utilities for uncertainty calibration validation.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_l1_data(csv_path="data/raw/train.csv"):
    """
    Load and preprocess Level 1 data from train.csv.
    
    Args:
        csv_path: Path to training CSV file
        
    Returns:
        DataFrame with Level 1 comparisons and preprocessed human labels
    """
    df = pd.read_csv(csv_path)
    
    # Filter Level 1 data (parent == 'ethereum')
    l1_df = df[df['parent'] == 'ethereum'].copy()
    
    # Preprocess human labels: 1.0->A, 2.0->B, multiplier<=1.2->Equal
    def convert_human_choice(row):
        if row['multiplier'] <= 1.2:
            return "Equal"
        elif row['choice'] == 1.0:
            return "A"
        elif row['choice'] == 2.0:
            return "B"
        else:
            return "Equal"
    
    l1_df['human_label'] = l1_df.apply(convert_human_choice, axis=1)
    
    logger.info(f"Loaded {len(l1_df)} Level 1 comparisons")
    logger.info(f"Human label distribution: {l1_df['human_label'].value_counts().to_dict()}")
    
    return l1_df