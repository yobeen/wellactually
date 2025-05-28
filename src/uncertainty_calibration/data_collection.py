# src/uncertainty_calibration/data_collection.py
#!/usr/bin/env python3
"""
Data collection pipeline for uncertainty calibration.
Collects LLM responses across temperature sweeps for training data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import time
from dataclasses import dataclass

# Import existing components
import sys
sys.path.append('src')
from uncertainty_calibration.multi_model_engine import MultiModelEngine, ModelResponse
from uncertainty_calibration.level1_prompts import Level1PromptGenerator
from uncertainty_calibration.level2_prompts import Level2PromptGenerator  
from uncertainty_calibration.level3_prompts import Level3PromptGenerator
from uncertainty_calibration.model_metadata import (
    MODEL_PARAMS, TEMPERATURE_SWEEP, get_model_metadata, validate_model_id
)

logger = logging.getLogger(__name__)

@dataclass
class CalibrationDataPoint:
    """Single data point for calibration training."""
    question_id: str
    level: int
    model_id: str
    temperature: float
    raw_uncertainty: float
    model_prediction: str
    correct_answer: str
    is_correct: bool
    param_count: float
    confidence_score: Optional[float] = None
    logprobs: Optional[Dict] = None

class UncertaintyDataCollector:
    """
    Collects LLM responses for uncertainty calibration training.
    """
    
    def __init__(self, config, models_subset: Optional[List[str]] = None):
        """Initialize data collector."""
        self.config = config
        self.engine = MultiModelEngine(config)
        
        # Initialize prompt generators
        self.level1_generator = Level1PromptGenerator(config)
        self.level2_generator = Level2PromptGenerator(config)
        self.level3_generator = Level3PromptGenerator(config)
        
        # Select models to use
        all_models = list(MODEL_PARAMS.keys())
        self.models = models_subset if models_subset else all_models[:6]  # Limit for efficiency
        
        # Validate models
        self.models = [m for m in self.models if validate_model_id(m)]
        
        logger.info(f"Initialized collector with {len(self.models)} models")
    
    def collect_training_data(self, train_df: pd.DataFrame, 
                            temperatures: List[float] = None,
                            max_samples_per_level: int = 50) -> List[CalibrationDataPoint]:
        """
        Collect training data by querying LLMs on train.csv examples.
        
        Args:
            train_df: Training dataframe (train.csv)
            temperatures: Temperature values to sweep
            max_samples_per_level: Limit samples per level for efficiency
            
        Returns:
            List of calibration data points
        """
        if temperatures is None:
            temperatures = TEMPERATURE_SWEEP
        
        data_points = []
        
        # Classify data by levels
        level1_data = train_df[train_df['parent'] == 'ethereum'].head(max_samples_per_level)
        level2_data = train_df[train_df['parent'] == 'originality'].head(max_samples_per_level)
        level3_data = train_df[~train_df['parent'].isin(['ethereum', 'originality'])].head(max_samples_per_level)
        
        # Process each level
        for level, level_data in [(1, level1_data), (2, level2_data), (3, level3_data)]:
            if len(level_data) == 0:
                continue
                
            logger.info(f"Processing Level {level}: {len(level_data)} samples")
            
            for idx, row in level_data.iterrows():
                question_id = f"level{level}_{idx}"
                
                # Generate prompt and get correct answer
                prompt, correct_answer = self._generate_prompt_and_answer(row, level)
                if not prompt:
                    continue
                
                # Query all models at all temperatures
                for model_id in self.models:
                    for temperature in temperatures:
                        try:
                            data_point = self._query_model_for_calibration(
                                question_id=question_id,
                                level=level,
                                model_id=model_id,
                                temperature=temperature,
                                prompt=prompt,
                                correct_answer=correct_answer
                            )
                            
                            if data_point:
                                data_points.append(data_point)
                                
                        except Exception as e:
                            logger.warning(f"Failed to query {model_id} at temp {temperature}: {e}")
                            continue
                
                # Rate limiting
                time.sleep(0.1)
        
        logger.info(f"Collected {len(data_points)} total data points")
        return data_points
    
    def _generate_prompt_and_answer(self, row: pd.Series, level: int) -> Tuple[List[Dict], str]:
        """Generate prompt and determine correct answer for a training row."""
        
        try:
            if level == 1:
                # Level 1: A vs B comparison
                repo_a = {"url": row['repo_a'], "name": row['repo_a'].split('/')[-1] if '/' in str(row['repo_a']) else str(row['repo_a'])}
                repo_b = {"url": row['repo_b'], "name": row['repo_b'].split('/')[-1] if '/' in str(row['repo_b']) else str(row['repo_b'])}
                
                prompt = self.level1_generator.create_comparison_prompt(repo_a, repo_b)
                
                # Map choice to answer
                choice = row['choice']
                if choice == 1.0:
                    correct_answer = "A"
                elif choice == 0.0:
                    correct_answer = "B"
                else:
                    correct_answer = "Equal"
                    
            elif level == 2:
                # Level 2: Originality assessment (1-10)
                repo = {"url": row['repo_a'], "name": row['repo_a'].split('/')[-1] if '/' in str(row['repo_a']) else str(row['repo_a'])}
                prompt = self.level2_generator.create_originality_prompt(repo)
                
                # Convert choice to bucket (assuming choice is 0-1, map to 1-10)
                choice = row['choice']
                correct_answer = str(max(1, min(10, int(choice * 10))))
                    
            elif level == 3:
                # Level 3: Dependency comparison
                dep_a = row['repo_a']
                dep_b = row['repo_b'] 
                parent = row['parent']
                
                prompt = self.level3_generator.create_dependency_comparison_prompt(dep_a, dep_b, parent)
                
                # Map choice to answer
                choice = row['choice']
                if choice == 1.0:
                    correct_answer = "A"
                elif choice == 0.0:
                    correct_answer = "B"
                else:
                    correct_answer = "Equal"
            else:
                return None, None
                
            return prompt, correct_answer
            
        except Exception as e:
            logger.error(f"Error generating prompt for level {level}: {e}")
            return None, None
    
    def _query_model_for_calibration(self, question_id: str, level: int, model_id: str,
                                   temperature: float, prompt: List[Dict], 
                                   correct_answer: str) -> Optional[CalibrationDataPoint]:
        """Query a single model and create calibration data point."""
        
        try:
            # Modify prompt for temperature
            modified_prompt = prompt.copy()
            
            # Query model with specific temperature
            responses = self.engine._query_single_model_with_temp(model_id, modified_prompt, temperature)
            
            if not responses.success:
                return None
            
            # Extract prediction and uncertainty
            prediction = responses.content.strip()
            
            # Calculate uncertainty from logprobs
            raw_uncertainty = self._calculate_uncertainty_from_logprobs(responses.logprobs, level)
            
            # Determine if correct
            is_correct = self._is_prediction_correct(prediction, correct_answer, level)
            
            # Get model metadata
            param_count = MODEL_PARAMS.get(model_id, 0.0)
            
            return CalibrationDataPoint(
                question_id=question_id,
                level=level,
                model_id=model_id,
                temperature=temperature,
                raw_uncertainty=raw_uncertainty,
                model_prediction=prediction,
                correct_answer=correct_answer,
                is_correct=is_correct,
                param_count=param_count,
                logprobs=responses.logprobs
            )
            
        except Exception as e:
            logger.error(f"Error querying model {model_id}: {e}")
            return None
    
    def _calculate_uncertainty_from_logprobs(self, logprobs: Dict, level: int) -> float:
        """Calculate uncertainty score from logprobs."""
        
        if not logprobs:
            return 1.0  # Maximum uncertainty if no logprobs
        
        try:
            if level in [1, 3]:  # A/B/Equal choices
                valid_tokens = ["A", "B", "Equal"]
            else:  # Level 2: 1-10 buckets
                valid_tokens = [str(i) for i in range(1, 11)]
            
            # Get probabilities for valid tokens
            token_probs = []
            for token in valid_tokens:
                prob = logprobs.get(token, 0.0)
                if prob > 0:
                    token_probs.append(prob)
            
            if not token_probs:
                return 1.0
            
            # Calculate entropy as uncertainty measure
            total_prob = sum(token_probs)
            if total_prob == 0:
                return 1.0
            
            normalized_probs = [p / total_prob for p in token_probs]
            entropy = -sum(p * np.log2(p + 1e-10) for p in normalized_probs if p > 0)
            
            # Normalize entropy to [0, 1]
            max_entropy = np.log2(len(valid_tokens))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return min(1.0, max(0.0, normalized_entropy))
            
        except Exception as e:
            logger.warning(f"Error calculating uncertainty: {e}")
            return 1.0
    
    def _is_prediction_correct(self, prediction: str, correct_answer: str, level: int) -> bool:
        """Check if model prediction matches correct answer."""
        
        prediction = prediction.strip().upper()
        correct_answer = correct_answer.strip().upper()
        
        if level in [1, 3]:  # A/B/Equal
            return prediction in correct_answer or correct_answer in prediction
        else:  # Level 2: numeric
            try:
                pred_num = int(''.join(filter(str.isdigit, prediction)))
                correct_num = int(correct_answer)
                return pred_num == correct_num
            except:
                return False
    
    def save_collected_data(self, data_points: List[CalibrationDataPoint], 
                          output_path: str) -> pd.DataFrame:
        """Save collected data to CSV for training."""
        
        rows = []
        for dp in data_points:
            rows.append({
                'question_id': dp.question_id,
                'level': dp.level,
                'model_id': dp.model_id,
                'temperature': dp.temperature,
                'raw_uncertainty': dp.raw_uncertainty,
                'model_prediction': dp.model_prediction,
                'correct_answer': dp.correct_answer,
                'is_correct': dp.is_correct,
                'param_count': dp.param_count
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} calibration data points to {output_path}")
        return df

# Extension to MultiModelEngine for temperature control
class MultiModelEngine:
    """Extended version with temperature control."""
    
    def _query_single_model_with_temp(self, model_id: str, prompt: List[Dict], temperature: float) -> ModelResponse:
        """Query single model with specific temperature."""
        
        # This would modify the existing multi_model_engine._query_single_model
        # to accept temperature parameter
        payload = {
            "model": model_id,
            "messages": prompt,
            "max_tokens": 10,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": 5
        }
        
        # Use existing logic from multi_model_engine._query_single_model
        # but with temperature override
        pass  # Implementation would adapt existing method