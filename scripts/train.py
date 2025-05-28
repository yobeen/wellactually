#!/usr/bin/env python3
"""
Main training script for Cross-Model Uncertainty Aggregation.
"""

import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    log.info("Starting Cross-Model Uncertainty Aggregation training...")
    log.info(f"Experiment: {cfg.experiment.name}")
    
    # TODO: Implement training pipeline
    log.info("Training pipeline not yet implemented")

if __name__ == "__main__":
    main()
