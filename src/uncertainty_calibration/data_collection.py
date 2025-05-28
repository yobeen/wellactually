# src/uncertainty_calibration/data_collection.py
"""
Data collection for uncertainty calibration training.
Implements temperature sweep protocol across multiple models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from llm_augmentation.engines.multi_model_engine import MultiModelEngine
from llm_augmentation.prompts.level1_prompts import Level1PromptGenerator
from llm_augmentation.prompts.level2_prompts import Level2PromptGenerator
from llm_augmentation.prompts.level3_prompts import Level3PromptGenerator
from uncertainty_calibration.model_metadata import get_model_params

class CalibrationDataCollector:
    """Collects responses across temperature range for calibration training."""
    
    def __init__(self, config):
        self.config = config
        self.engine = MultiModelEngine(config)
        self.temperatures = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Initialize prompt generators
        self.level1_prompts = Level1PromptGenerator(config)
        self.level2_prompts = Level2PromptGenerator(config)
        self.level3_prompts = Level3PromptGenerator(config)
    
    def collect_training_data(self, questions: List[Dict], models: List[str]) -> List[Dict]:
        """
        Collect responses across temperature range for training data.
        
        Args:
            questions: List of question dictionaries with 'question', 'correct_answer', 'type'
            models: List of model IDs to query
            
        Returns:
            List of response dictionaries
        """
        all_responses = []
        
        for question in questions:
            question_id = question.get('id', questions.index(question))
            question_type = question.get('type', 'level1')  # Default to level1
            
            for model_id in models:
                for temperature in self.temperatures:
                    # Generate prompt based on question type
                    prompt = self._generate_prompt(question, question_type)
                    
                    # Query model with specific temperature
                    response = self._query_model_with_temp(
                        model_id=model_id,
                        prompt=prompt,
                        temperature=temperature
                    )
                    
                    if response.success:
                        # Extract answer and calculate correctness
                        prediction = self._extract_prediction(response.content, question_type)
                        is_correct = self._is_correct(prediction, question['correct_answer'], question_type)
                        
                        # Create training example
                        training_example = {
                            'question_id': question_id,
                            'model_id': model_id,
                            'temperature': temperature,
                            'raw_uncertainty': self._extract_uncertainty(response),
                            'prediction': prediction,
                            'correct_answer': question['correct_answer'],
                            'is_correct': is_correct,
                            'param_count': get_model_params(model_id),
                            'question_type': question_type,
                            'response_content': response.content,
                            'logprobs': response.logprobs
                        }
                        
                        all_responses.append(training_example)
        
        return all_responses
    
    def _generate_prompt(self, question: Dict, question_type: str) -> List[Dict]:
        """Generate prompt based on question type."""
        if question_type == 'level1':
            # Repository comparison
            return self.level1_prompts.create_comparison_prompt(
                question['repo_a'], question['repo_b']
            )
        elif question_type == 'level2':
            # Originality assessment  
            return self.level2_prompts.create_originality_prompt(question['repo'])
        elif question_type == 'level3':
            # Dependency comparison
            return self.level3_prompts.create_dependency_comparison_prompt(
                question['dep_a'], question['dep_b'], question['parent']
            )
        else:
            # Generic Q&A prompt
            return [
                {"role": "system", "content": "Answer the question accurately."},
                {"role": "user", "content": question['question']}
            ]
    
    def _query_model_with_temp(self, model_id: str, prompt: List[Dict], temperature: float):
        """Query model with specific temperature."""
        # Temporarily override temperature in config
        original_temp = self.config.prompts.level_1.temperature
        self.config.prompts.level_1.temperature = temperature
        
        # Query single model
        response = self.engine._query_single_model(model_id, prompt)
        
        # Restore original temperature
        self.config.prompts.level_1.temperature = original_temp
        
        return response
    
    def _extract_uncertainty(self, response) -> float:
        """Extract uncertainty measure from response."""
        if response.logprobs:
            # Use negative log probability of chosen token as uncertainty
            probs = list(response.logprobs.values())
            if probs:
                max_prob = max(probs)
                return -np.log(max_prob + 1e-10)  # Add small epsilon
        
        # Fallback: use content length as proxy
        return len(response.content) / 100.0
    
    def _extract_prediction(self, content: str, question_type: str) -> str:
        """Extract model prediction from response content."""
        content = content.strip().upper()
        
        if question_type in ['level1', 'level3']:
            # A/B/Equal choices
            if 'A' in content and 'B' not in content:
                return 'A'
            elif 'B' in content and 'A' not in content:
                return 'B'
            elif 'EQUAL' in content:
                return 'Equal'
            else:
                return content[:10]  # First 10 chars as fallback
        
        elif question_type == 'level2':
            # Extract number 1-10
            for char in content:
                if char.isdigit():
                    num = int(char)
                    if 1 <= num <= 10:
                        return str(num)
            return '5'  # Default middle value
        
        else:
            return content[:50]  # First 50 chars
    
    def _is_correct(self, prediction: str, correct_answer: str, question_type: str) -> bool:
        """Check if prediction matches correct answer."""
        if question_type == 'level2':
            # For originality, use tolerance
            try:
                pred_num = float(prediction)
                correct_num = float(correct_answer)
                return abs(pred_num - correct_num) <= 1.0  # Within 1 bucket
            except:
                return prediction == correct_answer
        
        return prediction.strip().upper() == correct_answer.strip().upper()


def load_questions_from_csv(csv_path: str) -> List[Dict]:
    """Load questions from train.csv and convert to standard format."""
    df = pd.read_csv(csv_path)
    questions = []
    
    for _, row in df.iterrows():
        # Convert train.csv format to question format
        question = {
            'id': len(questions),
            'repo_a': {'url': row['repo_a'], 'name': row['repo_a'].split('/')[-1]},
            'repo_b': {'url': row['repo_b'], 'name': row['repo_b'].split('/')[-1]},
            'parent': row.get('parent', ''),
            'correct_answer': 'A' if row['choice'] < 0.5 else 'B' if row['choice'] > 0.5 else 'Equal',
            'type': 'level1',  # Assuming level1 type from train.csv
            'multiplier': row.get('multiplier', 1.0),
            'reasoning': row.get('reasoning', '')
        }
        questions.append(question)
    
    return questions


def collect_calibration_dataset(config, output_path: str):
    """Main function to collect full calibration dataset."""
    collector = CalibrationDataCollector(config)
    
    # Load questions from train.csv
    questions = load_questions_from_csv('train.csv')
    
    # Get model list from config
    models = list(config.models.primary_models.values())
    
    # Collect responses
    responses = collector.collect_training_data(questions, models)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(responses)
    df.to_csv(output_path, index=False)
    
    print(f"Collected {len(responses)} training examples")
    print(f"Saved to {output_path}")
    
    return df