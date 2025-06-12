# src/uncertainty_calibration/data_collection.py
#!/usr/bin/env python3
"""
Data collection pipeline for uncertainty calibration with cache support.
Collects LLM responses across temperature sweeps for training data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from src.shared.multi_model_engine import MultiModelEngine
from src.shared.response_parser import ModelResponse
from src.tasks.l1.level1_prompts import Level1PromptGenerator
from src.tasks.originality.level2_prompts import Level2PromptGenerator
from src.tasks.l3.level3_prompts import Level3PromptGenerator
from src.shared.model_metadata import get_model_metadata, validate_model_id

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
    human_choice: float
    human_multiplier: float
    repo_a: str
    repo_b: str
    parent: str

class UncertaintyDataCollector:
    """
    Collects LLM responses for uncertainty calibration training with caching support.
    """

    def __init__(self, config, models_subset: Optional[List[str]] = None):
        """Initialize data collector."""
        self.config = config
        self.engine = MultiModelEngine(config)

        # Initialize prompt generators
        self.level1_generator = Level1PromptGenerator(config)
        self.level2_generator = Level2PromptGenerator(config)
        self.level3_generator = Level3PromptGenerator()

        # Select models to use
        if models_subset:
            self.models = [m for m in models_subset if validate_model_id(m)]
        else:
            # Use models from config
            primary_models = list(config.models.primary_models.values())
            self.models = [m for m in primary_models if validate_model_id(m)]

        logger.info(f"Initialized collector with {len(self.models)} models: {self.models}")

    def collect_training_data(self, train_df: pd.DataFrame,
                              temperatures: List[float] = None,
                              max_samples_per_level: int = 30) -> List[CalibrationDataPoint]:
        """
        Collect training data by querying LLMs on train.csv examples.
        """
        if temperatures is None:
            temperatures = self.config.temperature_sweep.temperatures

        data_points = []

        # Process each row in train.csv
        for idx, row in train_df.head(max_samples_per_level * 3).iterrows():
            # Determine level and generate prompt
            level, prompt, correct_answer = self._process_training_row(row, idx)
            if not prompt:
                continue

            logger.info(f"Processing question {idx} (Level {level})")

            # Query all models at all temperatures
            for model_id in self.models:
                for temperature in temperatures:
                    try:
                        # Fixed: Remove the level parameter that doesn't exist in the method signature
                        response = self.engine.query_single_model_with_temperature(
                            model_id, prompt, temperature, max_tokens=20
                        )

                        if response.success:
                            data_point = self._create_calibration_datapoint(
                                row, idx, level, response, correct_answer
                            )
                            data_points.append(data_point)
                        else:
                            logger.warning(f"Failed query: {model_id} temp={temperature} error={response.error}")

                    except Exception as e:
                        logger.warning(f"Error querying {model_id} at temp {temperature}: {e}")

        logger.info(f"Collected {len(data_points)} total calibration data points")
        
        # Print cache statistics if cache is available
        if hasattr(self.engine, 'get_cache_stats'):
            try:
                cache_stats = self.engine.get_cache_stats()
                if cache_stats.get('enabled'):
                    logger.info(f"Cache statistics: {cache_stats['total_files']} files, "
                               f"{cache_stats['total_size_mb']:.2f} MB")
            except Exception as e:
                logger.debug(f"Could not get cache stats: {e}")
        
        return data_points

    def _process_training_row(self, row: pd.Series, idx: int) -> Tuple[int, List[Dict], str]:
        """Process a single training row to generate prompt and correct answer."""

        # Determine level based on parent field
        if row['parent'] == 'ethereum':
            level = 1
        elif row['parent'] == 'originality':
            level = 2
        else:
            level = 3

        try:
            if level == 1:
                # Level 1: Repository comparison
                repo_a = {"url": str(row['repo_a'])}
                repo_b = {"url": str(row['repo_b'])}

                prompt = self.level1_generator.create_comparison_prompt(repo_a, repo_b)
                correct_answer = self._convert_choice_to_answer(row['choice'])

            elif level == 2:
                # Level 2: Originality assessment
                repo = {"url": str(row['repo_a'])}
                prompt = self.level2_generator.create_originality_prompt(repo)

                # Convert choice to 1-10 scale
                choice_val = float(row['choice'])
                if choice_val == 1.0:
                    correct_answer = "5"  # Default mid-range for binary choice
                elif choice_val == 2.0:
                    correct_answer = "8"  # Higher originality
                else:
                    correct_answer = str(max(1, min(10, int(choice_val * 10))))

            elif level == 3:
                # Level 3: Dependency comparison
                dep_a = str(row['repo_a'])
                dep_b = str(row['repo_b'])
                parent = str(row['parent'])

                prompt = self.level3_generator.create_dependency_comparison_prompt(dep_a, dep_b, parent)
                correct_answer = self._convert_choice_to_answer(row['choice'])

            return level, prompt, correct_answer

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            return None, None, None

    def _convert_choice_to_answer(self, choice: float) -> str:
        """Convert numeric choice to A/B/Equal answer."""
        choice = float(choice)

        if choice == 1.0:
            return "A"
        elif choice == 2.0:
            return "B"
        else:
            return "Equal"

    def _create_calibration_datapoint(self, row: pd.Series, idx: int, level: int,
                                      response: ModelResponse, correct_answer: str) -> CalibrationDataPoint:
        """Create calibration data point from model response."""

        # Extract model prediction from raw_choice (already parsed with structured format)
        prediction = response.raw_choice.strip()

        # Determine if correct using the parsed prediction
        is_correct = self._is_prediction_correct(prediction, correct_answer, level)

        # Get model metadata
        metadata = get_model_metadata(response.model_id)
        param_count = metadata['param_count'] or 0.0

        return CalibrationDataPoint(
            question_id=f"q_{idx}",
            level=level,
            model_id=response.model_id,
            temperature=response.temperature,
            raw_uncertainty=response.uncertainty,
            model_prediction=prediction,
            correct_answer=correct_answer,
            is_correct=is_correct,
            param_count=param_count,
            human_choice=float(row['choice']),
            human_multiplier=float(row['multiplier']),
            repo_a=str(row['repo_a']),
            repo_b=str(row['repo_b']),
            parent=str(row['parent'])
        )

    def _is_prediction_correct(self, prediction: str, correct_answer: str, level: int) -> bool:
        """Check if model prediction matches correct answer."""

        prediction = prediction.strip().upper()
        correct_answer = correct_answer.strip().upper()

        if level in [1, 3]:  # A/B/Equal choices
            # Direct comparison since prediction should be parsed correctly
            return prediction == correct_answer
        else:  # Level 2: numeric
            try:
                # Both should be numbers for level 2
                pred_num = int(prediction) if prediction.isdigit() else None
                correct_num = int(correct_answer) if correct_answer.isdigit() else None
                
                if pred_num is not None and correct_num is not None:
                    return pred_num == correct_num
                return False
            except:
                return False

    def save_collected_data(self, data_points: List[CalibrationDataPoint],
                            output_path: str) -> pd.DataFrame:
        """Save collected data to CSV."""

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
                'param_count': dp.param_count,
                'human_choice': dp.human_choice,
                'human_multiplier': dp.human_multiplier,
                'repo_a': dp.repo_a,
                'repo_b': dp.repo_b,
                'parent': dp.parent
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved {len(df)} calibration data points to {output_path}")
        return df

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information from the engine."""
        if hasattr(self.engine, 'get_cache_info'):
            return self.engine.get_cache_info()
        return {"cache_available": False}

    def clear_cache(self, model_id: Optional[str] = None):
        """Clear cache through the engine."""
        if hasattr(self.engine, 'clear_cache'):
            self.engine.clear_cache(model_id)