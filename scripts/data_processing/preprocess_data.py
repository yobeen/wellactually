#!/usr/bin/env python3
"""
Data preprocessing script for train.csv and other raw data.
"""

import pandas as pd
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main data preprocessing function."""
    log.info("Starting data preprocessing...")
    
    # Load train.csv
    raw_data_path = Path(cfg.data.raw_dir) / "train.csv"
    if raw_data_path.exists():
        df = pd.read_csv(raw_data_path)
        log.info(f"Loaded train.csv with {len(df)} rows")
        
        # TODO: Implement preprocessing logic
        
        # Save processed data
        processed_path = Path(cfg.data.processed_dir) / "train_processed.csv"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        log.info(f"Saved processed data to {processed_path}")
    else:
        log.warning(f"train.csv not found at {raw_data_path}")

if __name__ == "__main__":
    main()
