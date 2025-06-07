#!/usr/bin/env python3
"""
GitHub Seed Repository File Downloader

Downloads all root-level files from GitHub repositories listed in configs/seed_repositories.yaml
and saves them to data/raw/seed_repos/{author}/{repo}/ (lowercase)

Usage:
    python scripts/download_seed_files.py

Optional environment variables:
    GITHUB_TOKEN: GitHub personal access token for higher rate limits
"""

import os
import yaml
import requests
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

class GitHubSeedDownloader:
    """Downloads root-level files from GitHub repositories."""
    
    def __init__(self, config_path: str = "configs/seed_repositories.yaml"):
        self.config_path = config_path
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.session = requests.Session()
        
        # Set up headers
        headers = {'User-Agent': 'GitHub-Seed-Downloader/1.0'}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
            print("✓ Using GitHub token for authentication")
        else:
            print("⚠ No GitHub token found. Rate limited to 60 requests/hour")
        
        self.session.headers.update(headers)
        
        # Create base output directory
        self.output_base = Path("data/raw/seed_repos")
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # Stats tracking
        self.stats = {
            'repos_processed': 0,
            'repos_failed': 0,
            'files_downloaded': 0,
            'files_failed': 0,
            'total_size': 0
        }
    
    def load_repositories(self) -> List[Dict[str, Any]]:
        """Load repository list from YAML config file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            repos = config.get('seed_repositories', [])
            print(f"✓ Loaded {len(repos)} repositories from {self.config_path}")
            return repos
            
        except FileNotFoundError:
            print(f"✗ Config file not found: {self.config_path}")
            return []
        except yaml.YAMLError as e:
            print(f"✗ Error parsing YAML: {e}")
            return []
    
    def parse_github_url(self, url: str) -> Tuple[str, str]:
        """Parse GitHub URL to extract owner and repository name."""
        # Handle URLs like https://github.com/owner/repo
        match = re.match(r'https://github\.com/([^/]+)/([^/]+)', url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {url}")
        
        owner, repo = match.groups()
        return owner, repo
    
    def sanitize_filename(self, name: str) -> str:
        """Sanitize filename for filesystem compatibility."""
        # Remove/replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        return sanitized
    
    def get_root_files(self, owner: str, repo: str) -> List[Dict[str, Any]]:
        """Get list of root-level files from GitHub repository."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
        
        try:
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 404:
                print(f"  ✗ Repository not found: {owner}/{repo}")
                return []
            elif response.status_code == 403:
                print(f"  ✗ Access denied or rate limited: {owner}/{repo}")
                return []
            elif response.status_code != 200:
                print(f"  ✗ API error {response.status_code}: {owner}/{repo}")
                return []
            
            contents = response.json()
            
            # Filter to files only (exclude directories)
            files = [item for item in contents if item.get('type') == 'file']
            print(f"  ✓ Found {len(files)} root-level files")
            
            return files
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Network error: {e}")
            return []
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return []
    
    def download_file(self, file_info: Dict[str, Any], output_dir: Path) -> bool:
        """Download a single file from GitHub."""
        filename = file_info['name']
        download_url = file_info['download_url']
        file_size = file_info.get('size', 0)
        
        # Skip very large files (>50MB)
        if file_size > 50 * 1024 * 1024:
            print(f"    ⚠ Skipping large file: {filename} ({file_size / (1024*1024):.1f}MB)")
            return False
        
        try:
            # Download file content
            response = self.session.get(download_url, timeout=60)
            response.raise_for_status()
            
            # Save to local file
            safe_filename = self.sanitize_filename(filename)
            output_path = output_dir / safe_filename
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"    ✓ Downloaded: {filename} ({len(response.content)} bytes)")
            self.stats['files_downloaded'] += 1
            self.stats['total_size'] += len(response.content)
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"    ✗ Download failed: {filename} - {e}")
            self.stats['files_failed'] += 1
            return False
        except Exception as e:
            print(f"    ✗ Unexpected error downloading {filename}: {e}")
            self.stats['files_failed'] += 1
            return False
    
    def process_repository(self, repo_info: Dict[str, Any]) -> bool:
        """Process a single repository - download all root files."""
        repo_url = repo_info.get('url', '')
        repo_name = repo_info.get('name', 'unknown')
        
        print(f"\n📁 Processing: {repo_name}")
        print(f"   URL: {repo_url}")
        
        try:
            # Parse GitHub URL
            owner, repo = self.parse_github_url(repo_url)
            
            # Create output directory: data/raw/seed_repos/author/repo (lowercase)
            safe_owner = self.sanitize_filename(owner.lower())
            safe_repo = self.sanitize_filename(repo.lower())
            output_dir = self.output_base / safe_owner / safe_repo
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   Output: {output_dir}")
            
            # Get list of root files
            files = self.get_root_files(owner, repo)
            if not files:
                self.stats['repos_failed'] += 1
                return False
            
            # Download each file
            success_count = 0
            for file_info in files:
                if self.download_file(file_info, output_dir):
                    success_count += 1
                
                # Small delay to be respectful to GitHub
                time.sleep(0.1)
            
            print(f"  ✓ Repository complete: {success_count}/{len(files)} files downloaded")
            self.stats['repos_processed'] += 1
            return True
            
        except ValueError as e:
            print(f"  ✗ {e}")
            self.stats['repos_failed'] += 1
            return False
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            self.stats['repos_failed'] += 1
            return False
    
    def run(self):
        """Main execution function."""
        print("🚀 GitHub Seed Repository Downloader")
        print("=" * 50)
        
        # Load repository configuration
        repos = self.load_repositories()
        if not repos:
            print("No repositories to process. Exiting.")
            return
        
        print(f"📋 Processing {len(repos)} repositories...")
        
        # Process each repository
        for i, repo_info in enumerate(repos, 1):
            print(f"\n[{i}/{len(repos)}] ", end="")
            self.process_repository(repo_info)
            # Add delay between repositories to respect rate limits
            if i < len(repos):  # Don't sleep after the last repo
                time.sleep(1)
        
        # Print final statistics
        print("\n" + "=" * 50)
        print("📊 Download Summary:")
        print(f"   Repositories processed: {self.stats['repos_processed']}")
        print(f"   Repositories failed: {self.stats['repos_failed']}")
        print(f"   Files downloaded: {self.stats['files_downloaded']}")
        print(f"   Files failed: {self.stats['files_failed']}")
        print(f"   Total size: {self.stats['total_size'] / (1024*1024):.2f} MB")
        print(f"   Output directory: {self.output_base.absolute()}")
        
        if self.stats['repos_processed'] > 0:
            print("\n✅ Download completed successfully!")
        else:
            print("\n❌ No repositories were successfully processed.")

def main():
    """Main entry point."""
    downloader = GitHubSeedDownloader()
    downloader.run()

if __name__ == "__main__":
    main()