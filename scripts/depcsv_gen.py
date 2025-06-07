#!/usr/bin/env python3
"""
Dependencies CSV Generator Script

Transforms DeepFunding Repos Enhanced via OpenQ ENHANCED TEAMS.csv
into dependencies.csv format for Level 3 prompts generator.

Usage: python generate_dependencies.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_paths():
    """Setup input and output file paths."""
    input_file = "data/raw/DeepFunding Repos Enhanced via OpenQ - ENHANCED TEAMS.csv"
    output_file = "dependencies.csv"
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    return input_file, output_file

def load_and_validate_data(input_file):
    """Load and validate the input CSV data."""
    try:
        logger.info(f"Loading data from {input_file}")
        
        # Load with robust parsing
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # Strip whitespace from headers
        df.columns = df.columns.str.strip()
        
        logger.info(f"Loaded {len(df)} repositories with columns: {list(df.columns)}")
        
        # Validate required columns exist
        required_cols = ['githubLink', 'name', 'description']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def categorize_repository(row):
    """Categorize repository based on languages, description, and other features."""
    
    # Get relevant data with safe defaults
    name = str(row.get('name', '')).lower()
    description = str(row.get('description', '')).lower()
    languages = str(row.get('languages', '')).lower()
    dependencies = str(row.get('dependencies', '')).lower()
    
    # Define category keywords with flexible mapping
    category_keywords = {
        'web3_library': ['web3', 'ethereum', 'blockchain', 'dapp', 'defi', 'nft', 'smart contract'],
        'development_framework': ['framework', 'scaffold', 'template', 'boilerplate', 'starter'],
        'development_tool': ['tool', 'compiler', 'debugger', 'testing', 'deployment', 'cli'],
        'smart_contract': ['solidity', 'smart contract', 'contract', 'vyper'],
        'client': ['client', 'node', 'consensus', 'execution', 'validator'],
        'infrastructure': ['infrastructure', 'protocol', 'network', 'p2p', 'consensus'],
        'data_analysis': ['analysis', 'analytics', 'metrics', 'monitoring', 'dashboard'],
        'security_tool': ['security', 'audit', 'vulnerability', 'scanner'],
        'wallet': ['wallet', 'keystore', 'keys', 'signing'],
        'api_service': ['api', 'service', 'server', 'backend', 'microservice'],
        'web_library': ['react', 'vue', 'angular', 'frontend', 'ui', 'component'],
        'mobile_app': ['mobile', 'android', 'ios', 'app', 'flutter'],
        'documentation': ['docs', 'documentation', 'guide', 'tutorial'],
        'utility_library': ['utility', 'utils', 'helper', 'common', 'shared']
    }
    
    # Check description and name for category keywords
    text_to_check = f"{name} {description}"
    
    for category, keywords in category_keywords.items():
        if any(keyword in text_to_check for keyword in keywords):
            return category
    
    # Language-based fallback categorization
    if 'solidity' in languages:
        return 'smart_contract'
    elif any(lang in languages for lang in ['javascript', 'typescript', 'react', 'vue']):
        return 'web_library'
    elif any(lang in languages for lang in ['python', 'rust', 'go', 'java']):
        return 'general_library'
    elif 'dart' in languages or 'flutter' in languages:
        return 'mobile_library'
    else:
        return 'general_library'

def derive_primary_function(row):
    """Derive primary function from description and languages."""
    
    description = str(row.get('description', ''))
    languages = str(row.get('languages', ''))
    category = row.get('category', '')
    
    # Clean and truncate description
    cleaned_desc = re.sub(r'[^\w\s\-\.]', ' ', description)
    cleaned_desc = ' '.join(cleaned_desc.split()[:15])  # First 15 words
    
    if not cleaned_desc or cleaned_desc.lower() in ['nan', 'none', '']:
        # Generate function based on category and languages
        if category == 'web3_library':
            return f"Web3 integration and blockchain interaction"
        elif category == 'smart_contract':
            return f"Smart contract development and deployment"
        elif category == 'development_tool':
            return f"Development tooling and automation"
        elif languages:
            primary_lang = languages.split(',')[0].strip() if ',' in languages else languages
            return f"{primary_lang.title()} development utilities"
        else:
            return "General-purpose software library"
    
    return cleaned_desc

def derive_integration_patterns(row):
    """Derive integration patterns from dependencies and repository characteristics."""
    
    dependencies = str(row.get('dependencies', ''))
    category = row.get('category', '')
    languages = str(row.get('languages', ''))
    
    patterns = []
    
    # Analyze dependencies column
    if 'npm' in dependencies.lower() or 'node' in dependencies.lower():
        patterns.append("NPM package integration")
    if 'pip' in dependencies.lower() or 'python' in dependencies.lower():
        patterns.append("Python package integration")
    if 'cargo' in dependencies.lower() or 'rust' in dependencies.lower():
        patterns.append("Rust crate integration")
    if 'maven' in dependencies.lower() or 'gradle' in dependencies.lower():
        patterns.append("Java dependency management")
    
    # Category-based patterns
    if category == 'web3_library':
        patterns.append("Web3 provider integration")
    elif category == 'smart_contract':
        patterns.append("Smart contract compilation and deployment")
    elif category == 'development_framework':
        patterns.append("Framework scaffolding and templates")
    elif category == 'api_service':
        patterns.append("REST/GraphQL API integration")
    elif category == 'web_library':
        patterns.append("Frontend component integration")
    
    # Language-based patterns
    if 'javascript' in languages.lower() or 'typescript' in languages.lower():
        if 'import' not in ' '.join(patterns).lower():
            patterns.append("ES6 module imports")
    elif 'python' in languages.lower():
        if 'import' not in ' '.join(patterns).lower():
            patterns.append("Python import statements")
    
    # Default pattern if none identified
    if not patterns:
        patterns.append("Standard library integration")
    
    return "; ".join(patterns[:3])  # Limit to 3 patterns for readability

def transform_data(df):
    """Transform the dataframe to match dependencies.csv schema."""
    
    logger.info("Transforming data to dependencies.csv schema")
    
    # Create new dataframe with required columns
    transformed_df = pd.DataFrame()
    
    # Direct mappings
    transformed_df['dependency_url'] = df['githubLink']
    transformed_df['name'] = df['name']
    transformed_df['description'] = df['description'].fillna('')
    
    # Derive category for each repository
    logger.info("Categorizing repositories...")
    transformed_df['category'] = df.apply(categorize_repository, axis=1)
    
    # Derive primary function
    logger.info("Deriving primary functions...")
    transformed_df['primary_function'] = df.apply(
        lambda row: derive_primary_function({**row, 'category': transformed_df.loc[row.name, 'category']}), 
        axis=1
    )
    
    # Derive integration patterns
    logger.info("Deriving integration patterns...")
    transformed_df['integration_patterns'] = df.apply(
        lambda row: derive_integration_patterns({**row, 'category': transformed_df.loc[row.name, 'category']}), 
        axis=1
    )
    
    # Optional columns with useful information
    transformed_df['performance_characteristics'] = df.apply(
        lambda row: f"Activity: {row.get('activity', 'unknown')}, Stars: {row.get('totalStars', 0)}, Commits: {row.get('commitCount', 0)}", 
        axis=1
    )
    
    # Languages as is (not split as requested)
    transformed_df['languages'] = df['languages'].fillna('')
    
    # Additional metadata that might be useful
    transformed_df['reputation_score'] = df['reputation'].fillna(0)
    transformed_df['popularity_score'] = df['popularity'].fillna(0)
    
    return transformed_df

def validate_output(df):
    """Validate the transformed dataframe."""
    
    logger.info("Validating transformed data")
    
    # Check required columns
    required_columns = ['dependency_url', 'name', 'description', 'category', 'primary_function', 'integration_patterns']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in output: {missing_columns}")
    
    # Check for empty URLs
    empty_urls = df['dependency_url'].isna().sum()
    if empty_urls > 0:
        logger.warning(f"Found {empty_urls} repositories with missing URLs")
    
    # Log category distribution
    category_counts = df['category'].value_counts()
    logger.info(f"Category distribution:\n{category_counts}")
    
    # Check data quality
    logger.info(f"Total repositories: {len(df)}")
    logger.info(f"Repositories with descriptions: {(df['description'] != '').sum()}")
    logger.info(f"Unique categories: {df['category'].nunique()}")
    
    return True

def save_dependencies_csv(df, output_file):
    """Save the transformed dataframe to CSV."""
    
    try:
        logger.info(f"Saving dependencies to {output_file}")
        
        # Save with proper encoding
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        logger.info(f"Successfully saved {len(df)} dependencies to {output_file}")
        
        # Show sample of output
        logger.info("Sample of generated dependencies:")
        sample_df = df[['name', 'category', 'primary_function']].head(5)
        for _, row in sample_df.iterrows():
            logger.info(f"  {row['name']} [{row['category']}]: {row['primary_function'][:50]}...")
            
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        raise

def main():
    """Main execution function."""
    
    logger.info("Starting dependencies CSV generation")
    
    try:
        # Setup paths
        input_file, output_file = setup_paths()
        
        # Load and validate input data
        df = load_and_validate_data(input_file)
        
        # Transform data
        transformed_df = transform_data(df)
        
        # Validate output
        validate_output(transformed_df)
        
        # Save to CSV
        save_dependencies_csv(transformed_df, output_file)
        
        logger.info("Dependencies CSV generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise

if __name__ == "__main__":
    main()