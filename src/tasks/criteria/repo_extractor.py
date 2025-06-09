# src/uncertainty_calibration/criteria_assessment/repo_extractor.py
"""
Repository extractor for criteria assessment.
Extracts unique repositories from train.csv for criteria-based evaluation.
"""

import pandas as pd
import logging
from typing import List, Set
from urllib.parse import urlparse
import re

logger = logging.getLogger(__name__)

class RepositoryExtractor:
    """
    Extracts and validates unique repositories from training data.
    """
    
    def __init__(self):
        """Initialize the repository extractor."""
        self.github_pattern = re.compile(r'https://github\.com/[^/]+/[^/]+')
        
    def extract_unique_repos(self, csv_path: str = "data/raw/train.csv") -> List[str]:
        """
        Extract unique repository URLs from train.csv.
        
        Args:
            csv_path: Path to training CSV file
            
        Returns:
            List of unique, validated repository URLs
        """
        try:
            # Load training data
            df = pd.read_csv(csv_path).head(1)
            logger.info(f"Loaded {len(df)} rows from {csv_path}")
            
            # Extract all repository URLs
            all_repos = set()
            
            # Add repos from repo_a column
            if 'repo_a' in df.columns:
                repo_a_urls = df['repo_a'].dropna().unique()
                all_repos.update(repo_a_urls)
                
            # Add repos from repo_b column  
            if 'repo_b' in df.columns:
                repo_b_urls = df['repo_b'].dropna().unique()
                all_repos.update(repo_b_urls)
                
            logger.info(f"Found {len(all_repos)} total unique repository URLs")
            
            # Validate and clean URLs
            validated_repos = self._validate_repositories(all_repos)
            
            logger.info(f"Validated {len(validated_repos)} repository URLs")
            
            return sorted(validated_repos)
            
        except Exception as e:
            logger.error(f"Error extracting repositories from {csv_path}: {e}")
            raise
    
    def _validate_repositories(self, repo_urls: Set[str]) -> List[str]:
        """
        Validate and clean repository URLs.
        
        Args:
            repo_urls: Set of repository URLs to validate
            
        Returns:
            List of validated repository URLs
        """
        validated = []
        
        for url in repo_urls:
            if not url or not isinstance(url, str):
                continue
                
            # Clean URL
            cleaned_url = url.strip()
            
            # Basic GitHub URL validation
            if self._is_valid_github_url(cleaned_url):
                validated.append(cleaned_url)
            else:
                logger.warning(f"Invalid repository URL: {url}")
                
        return validated
    
    def _is_valid_github_url(self, url: str) -> bool:
        """
        Check if URL is a valid GitHub repository URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid GitHub repo URL
        """
        try:
            # Check basic pattern
            if not self.github_pattern.match(url):
                return False
                
            # Parse URL
            parsed = urlparse(url)
            
            # Must be github.com
            if parsed.netloc != 'github.com':
                return False
                
            # Must have owner/repo format
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 2:
                return False
                
            # Owner and repo name should not be empty
            if not path_parts[0] or not path_parts[1]:
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_repo_info(self, repo_url: str) -> dict:
        """
        Extract repository information from URL.
        
        Args:
            repo_url: Repository URL
            
        Returns:
            Dictionary with repo info
        """
        try:
            parsed = urlparse(repo_url)
            path_parts = parsed.path.strip('/').split('/')
            
            return {
                'url': repo_url,
                'owner': path_parts[0],
                'name': path_parts[1],
                'full_name': f"{path_parts[0]}/{path_parts[1]}"
            }
            
        except Exception as e:
            logger.warning(f"Error parsing repo info from {repo_url}: {e}")
            return {
                'url': repo_url,
                'owner': 'unknown',
                'name': 'unknown', 
                'full_name': 'unknown/unknown'
            }