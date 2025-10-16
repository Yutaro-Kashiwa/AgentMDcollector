#!/usr/bin/env python3
"""
High-performance GitHub collector using multiple tokens in parallel.
Optimizes collection speed by distributing searches across token pool.
"""

import os
import sys
import time
import json
import pickle
import threading
import queue
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dataclasses import dataclass
from threading import Lock, Event

@dataclass
class TokenInfo:
    """Track token state and rate limits"""
    token: str
    id: int
    search_remaining: int = 30
    search_reset: datetime = None
    core_remaining: int = 5000
    core_reset: datetime = None
    is_available: bool = True
    repos_collected: int = 0
    errors: int = 0
    last_used: datetime = None

class MultiTokenCollector:
    def __init__(self, tokens: List[str]):
        """Initialize with multiple GitHub tokens"""
        self.tokens = [TokenInfo(token=t, id=i+1) for i, t in enumerate(tokens)]
        self.token_lock = Lock()
        self.results_lock = Lock()
        
        self.collected_repos = {}  # {url: repo_info}
        self.file_queue = queue.Queue()

        # Create outputs directory
        self.output_dir = 'outputs'
        os.makedirs(self.output_dir, exist_ok=True)

        # Output file paths
        self.results_file = os.path.join(self.output_dir, 'multi_token_results.json')
        self.urls_file = os.path.join(self.output_dir, 'multi_token_urls.txt')
        self.state_file = os.path.join(self.output_dir, 'multi_token_state.pkl')
        self.stats_file = os.path.join(self.output_dir, 'collection_statistics.json')

        # Comprehensive AI file list
        self.target_files = [
            # Claude/Anthropic
            'Claude.md', 'CLAUDE.md',
            
            # Agents
            'Agent.md', 'AGENT.md',
            
            # # OpenAI/ChatGPT
            # 'openai.md', 'chatgpt.md',
            #
            # # GitHub Copilot
            # '.copilot', 'copilot.json', 'copilot.yaml', '.copilotignore',
            # 'copilot.config.json', '.github/copilot.yml',
            #
            # # AI/LLM Documentation
            # 'AI.md', 'LLM.md', 'PROMPTS.md', 'PROMPT.md', 'prompts.json',
            # 'prompts.yaml', 'prompt.json', 'prompt.yaml',
            #
            # # Cursor IDE
            # '.cursorrules', 'cursor.json', '.cursor', 'cursor.config.json',
            # '.cursorignore',
            #
            # # Other AI Assistants
            # 'bard.json', 'bard.yaml', '.bard',
            # 'gemini.json', 'gemini.yaml', '.gemini',
            # 'llama.json', 'llama.yaml', '.llama',
            # 'palm.json', 'palm.yaml', '.palm',
            #
            # # AI Configuration
            # '.ai', 'ai.json', 'ai.yaml', 'ai-config.json', 'ai-config.yaml',
            # 'ai.config.json', 'ai.config.yaml', '.aiconfig',
            #
            # # LLM Configuration
            # 'llm.json', 'llm.yaml', 'llm-config.json', 'llm-config.yaml',
            # 'llm.config.json', '.llmconfig',
            #
            # # Model Configuration
            # 'models.json', 'models.yaml', 'model.json', 'model.yaml',
            # 'model.config.json', 'models.config.json',
            #
            # # Training/Fine-tuning
            # 'training.json', 'training.yaml', 'train.json', 'train.yaml',
            # 'finetune.json', 'finetune.yaml', 'fine-tune.json', 'fine-tune.yaml',
            # 'dataset.json', 'dataset.yaml',
            #
            # # Embeddings/Vectors
            # 'embeddings.json', 'embeddings.yaml', 'vectors.json', 'vectors.yaml',
            # 'embed.json', 'embed.yaml', 'embedding.config.json',
            #
            # # Assistant/Bot Configuration
            # 'assistant.json', 'assistant.yaml', '.assistant',
            # 'bot.json', 'bot.yaml', '.bot', 'chatbot.json', 'chatbot.yaml',
            #
            # # Prompt Engineering
            # 'promptflow.json', 'promptflow.yaml',
            # 'prompt-engineering.md', 'prompt_engineering.md',
            #
            # # RAG Configuration
            # 'rag.json', 'rag.yaml', 'rag.config.json',
            # 'retrieval.json', 'retrieval.yaml',
            #
            # # Vector Stores
            # 'pinecone.json', 'pinecone.yaml',
            # 'weaviate.json', 'weaviate.yaml',
            # 'chroma.json', 'chroma.yaml',
            #
            # # LangChain
            # 'langchain.json', 'langchain.yaml', '.langchain',
            # 'chains.json', 'chains.yaml'
        ]
        
        self.base_url = 'https://api.github.com'
        self.stop_event = Event()
        
        # Load previous state if exists
        self.load_state()
        
        # Initialize token states
        self.update_all_token_limits()
    
    def load_state(self):
        """Load previous collection state"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'rb') as f:
                    state = pickle.load(f)
                    self.collected_repos = state.get('repos', {})
                    print(f"✅ Loaded state: {len(self.collected_repos)} repos already collected")
                    return state.get('completed_files', set())
            except:
                pass
        return set()
    
    def save_state(self):
        """Save current collection state"""
        with self.results_lock:
            state = {
                'repos': self.collected_repos,
                'timestamp': datetime.now().isoformat(),
                'completed_files': self.get_completed_files()
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
    
    def get_completed_files(self) -> Set[str]:
        """Track which files have been fully searched"""
        # This would need more sophisticated tracking in production
        return set()
    
    def update_token_limits(self, token_info: TokenInfo):
        """Update rate limits for a specific token"""
        headers = {
            'Authorization': f'token {token_info.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(f'{self.base_url}/rate_limit', headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                search = data['resources']['search']
                token_info.search_remaining = search['remaining']
                token_info.search_reset = datetime.fromtimestamp(search['reset'])
                
                core = data['resources']['core']
                token_info.core_remaining = core['remaining']
                token_info.core_reset = datetime.fromtimestamp(core['reset'])
                
                token_info.is_available = token_info.search_remaining > 0
                
                return True
        except Exception as e:
            print(f"❌ Token {token_info.id} rate limit check failed: {e}")
            token_info.errors += 1
            if token_info.errors > 5:
                token_info.is_available = False
        
        return False
    
    def update_all_token_limits(self):
        """Update rate limits for all tokens"""
        print("\n📊 Checking token rate limits...")
        
        for token_info in self.tokens:
            self.update_token_limits(token_info)
            print(f"  Token {token_info.id}: {token_info.search_remaining}/30 searches, "
                  f"Reset: {token_info.search_reset.strftime('%H:%M:%S') if token_info.search_reset else 'Unknown'}")
    
    def get_available_token(self) -> Optional[TokenInfo]:
        """Get next available token with remaining rate limit"""
        with self.token_lock:
            # First try tokens with remaining searches
            available = [t for t in self.tokens if t.is_available and t.search_remaining > 0]
            
            if available:
                # Use token with most remaining requests
                token = max(available, key=lambda t: t.search_remaining)
                token.last_used = datetime.now()
                return token
            
            # Check if any token will reset soon
            next_reset = None
            for token in self.tokens:
                if token.search_reset:
                    if not next_reset or token.search_reset < next_reset:
                        next_reset = token.search_reset
            
            if next_reset:
                wait_time = (next_reset - datetime.now()).total_seconds() + 5
                if wait_time > 0 and wait_time < 3600:  # Wait up to 1 hour
                    print(f"\n⏳ All tokens exhausted. Waiting {wait_time/60:.1f} minutes until reset...")
                    time.sleep(wait_time)
                    self.update_all_token_limits()
                    return self.get_available_token()
            
            return None
    
    def search_file(self, filename: str, token_info: TokenInfo) -> Tuple[int, List[Dict]]:
        """Search for repositories with a specific file using given token"""
        headers = {
            'Authorization': f'token {token_info.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        repos = []
        search_query = f'filename:{filename} path:/'
        page = 1
        max_pages = 10  # GitHub limits to 1000 results
        
        print(f"🔍 Token {token_info.id} searching: {filename}")
        
        while page <= max_pages and token_info.search_remaining > 0:
            url = f'{self.base_url}/search/code'
            params = {
                'q': search_query,
                'per_page': 100,
                'page': page
            }
            
            try:
                response = requests.get(url, headers=headers, params=params, timeout=30)
                
                # Update remaining requests
                if 'X-RateLimit-Remaining' in response.headers:
                    token_info.search_remaining = int(response.headers['X-RateLimit-Remaining'])
                
                if response.status_code == 403:
                    # Rate limit hit
                    token_info.is_available = False
                    token_info.search_remaining = 0
                    break
                
                if response.status_code == 422:
                    # Search not supported for this file
                    break
                
                response.raise_for_status()
                data = response.json()
                
                if 'items' not in data or len(data['items']) == 0:
                    break
                
                for item in data['items']:
                    repo_info = {
                        'name': item['repository']['full_name'],
                        'url': item['repository']['html_url'],
                        'file': filename,
                        'description': item['repository'].get('description', ''),
                        'stars': item['repository'].get('stargazers_count', 0),
                        'forks': item['repository'].get('forks_count', 0),
                        'language': item['repository'].get('language', 'Unknown'),
                        'created_at': item['repository'].get('created_at', ''),
                        'updated_at': item['repository'].get('updated_at', ''),
                        'topics': item['repository'].get('topics', []),
                        'collected_at': datetime.now().isoformat(),
                        'collected_by_token': token_info.id
                    }
                    repos.append(repo_info)
                
                if page * 100 >= 1000:
                    break
                
                page += 1
                time.sleep(1)  # Respectful delay
                
            except requests.exceptions.RequestException as e:
                print(f"  ❌ Token {token_info.id} error: {e}")
                token_info.errors += 1
                break
        
        token_info.repos_collected += len(repos)
        return token_info.id, repos
    
    def worker_thread(self, worker_id: int):
        """Worker thread that processes files from queue"""
        while not self.stop_event.is_set():
            try:
                # Get file from queue
                filename = self.file_queue.get(timeout=1)
                
                # Get available token
                token = self.get_available_token()
                if not token:
                    print(f"Worker {worker_id}: No tokens available")
                    self.file_queue.put(filename)  # Put back in queue
                    time.sleep(10)
                    continue
                
                # Search for file
                token_id, repos = self.search_file(filename, token)
                
                # Store results
                with self.results_lock:
                    new_repos = 0
                    for repo in repos:
                        repo_url = repo['url']
                        if repo_url not in self.collected_repos:
                            self.collected_repos[repo_url] = repo
                            self.collected_repos[repo_url]['files_found'] = [filename]
                            new_repos += 1
                        else:
                            # Add file to existing repo
                            if 'files_found' not in self.collected_repos[repo_url]:
                                self.collected_repos[repo_url]['files_found'] = []
                            if filename not in self.collected_repos[repo_url]['files_found']:
                                self.collected_repos[repo_url]['files_found'].append(filename)
                
                print(f"  ✅ Token {token_id} found {new_repos} new repos for {filename}")
                
                self.file_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    def collect_parallel(self):
        """Main parallel collection process"""
        print("=" * 80)
        print("🚀 Multi-Token Parallel GitHub Collector")
        print("=" * 80)
        print(f"🔑 Active tokens: {len([t for t in self.tokens if t.is_available])}/{len(self.tokens)}")
        print(f"📁 Target files: {len(self.target_files)}")
        print(f"🔄 Previously collected: {len(self.collected_repos)} repos")
        print("-" * 80)
        
        # Fill queue with files to search
        completed_files = self.load_state()
        for filename in self.target_files:
            if filename not in completed_files:
                self.file_queue.put(filename)
        
        print(f"📦 Files to process: {self.file_queue.qsize()}")
        
        # Start worker threads (one per token for maximum parallelism)
        num_workers = min(len(self.tokens), self.file_queue.qsize())
        workers = []
        
        print(f"🚀 Starting {num_workers} worker threads...")
        
        for i in range(num_workers):
            worker = threading.Thread(target=self.worker_thread, args=(i+1,))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Monitor progress
        start_time = datetime.now()
        last_save = datetime.now()
        
        try:
            while not self.file_queue.empty():
                # Progress update
                remaining = self.file_queue.qsize()
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if elapsed > 0:
                    rate = (len(self.target_files) - remaining) / (elapsed / 60)
                    eta = remaining / rate if rate > 0 else 0
                    
                    print(f"\n📊 Progress: {len(self.target_files) - remaining}/{len(self.target_files)} files")
                    print(f"⏱️  Rate: {rate:.1f} files/min | ETA: {eta:.1f} minutes")
                    print(f"🗂️  Unique repos collected: {len(self.collected_repos)}")
                
                # Token status
                active_tokens = [t for t in self.tokens if t.is_available]
                print(f"🔑 Active tokens: {len(active_tokens)}/{len(self.tokens)}")
                
                # Periodic save
                if (datetime.now() - last_save).total_seconds() > 60:
                    self.save_state()
                    self.save_results()
                    last_save = datetime.now()
                
                # Periodic token limit update
                if elapsed > 0 and int(elapsed) % 300 == 0:  # Every 5 minutes
                    self.update_all_token_limits()
                
                time.sleep(10)
            
            # Wait for workers to finish
            self.file_queue.join()
            
        except KeyboardInterrupt:
            print("\n⚠️  Collection interrupted by user")
        
        finally:
            # Stop workers
            self.stop_event.set()
            for worker in workers:
                worker.join(timeout=5)
            
            # Save final results
            self.save_state()
            self.save_results()
            self.generate_statistics()
    
    def save_results(self):
        """Save collected repository data"""
        repos_list = list(self.collected_repos.values())
        
        # Sort by stars
        repos_list.sort(key=lambda x: x.get('stars', 0), reverse=True)
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_repositories': len(repos_list),
            'tokens_used': len(self.tokens),
            'files_searched': len(self.target_files),
            'repositories': repos_list
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Save URLs
        with open(self.urls_file, 'w') as f:
            for repo in repos_list:
                f.write(f"{repo['url']}\n")
    
    def generate_statistics(self):
        """Generate detailed collection statistics"""
        repos_list = list(self.collected_repos.values())
        
        stats = {
            'collection_date': datetime.now().isoformat(),
            'total_repositories': len(repos_list),
            'total_tokens': len(self.tokens),
            'files_searched': len(self.target_files),
            'token_performance': {},
            'language_distribution': {},
            'file_distribution': {},
            'top_topics': {},
            'repos_with_multiple_files': 0
        }
        
        # Token performance
        for token in self.tokens:
            stats['token_performance'][f'token_{token.id}'] = {
                'repos_collected': token.repos_collected,
                'errors': token.errors,
                'remaining_searches': token.search_remaining
            }
        
        # Language distribution
        languages = {}
        topics_count = {}
        file_count = {}
        multi_file_repos = 0
        
        for repo in repos_list:
            # Language stats
            lang = repo.get('language', 'Unknown')
            languages[lang] = languages.get(lang, 0) + 1
            
            # File stats
            files = repo.get('files_found', [])
            if len(files) > 1:
                multi_file_repos += 1
            
            for file in files:
                file_count[file] = file_count.get(file, 0) + 1
            
            # Topic stats
            for topic in repo.get('topics', []):
                topics_count[topic] = topics_count.get(topic, 0) + 1
        
        stats['language_distribution'] = dict(sorted(languages.items(), 
                                                    key=lambda x: x[1], 
                                                    reverse=True)[:20])
        stats['file_distribution'] = dict(sorted(file_count.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True)[:20])
        stats['top_topics'] = dict(sorted(topics_count.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:20])
        stats['repos_with_multiple_files'] = multi_file_repos
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 COLLECTION COMPLETE - STATISTICS")
        print("=" * 80)
        print(f"✅ Total repositories: {len(repos_list)}")
        print(f"📁 Files searched: {len(self.target_files)}")
        print(f"🔑 Tokens used: {len(self.tokens)}")
        print(f"🔄 Repos with multiple AI files: {multi_file_repos}")
        
        print("\n🏆 Top 10 repositories by stars:")
        for i, repo in enumerate(repos_list[:10], 1):
            print(f"{i:2}. ⭐ {repo.get('stars', 0):,} - {repo['name']}")
            files = repo.get('files_found', [])
            if files:
                print(f"    📁 Files: {', '.join(files[:5])}")
        
        print("\n🌐 Top 5 programming languages:")
        for lang, count in list(stats['language_distribution'].items())[:5]:
            print(f"  • {lang}: {count} repos")
        
        print("\n🏷️ Top 5 topics:")
        for topic, count in list(stats['top_topics'].items())[:5]:
            print(f"  • {topic}: {count} repos")
        
        print("\n🔑 Token performance:")
        total_collected = sum(t.repos_collected for t in self.tokens)
        for token in self.tokens:
            pct = (token.repos_collected / total_collected * 100) if total_collected > 0 else 0
            print(f"  Token {token.id}: {token.repos_collected} repos ({pct:.1f}%), "
                  f"{token.errors} errors")
        
        print(f"\n💾 Output files:")
        print(f"  • Results: {self.results_file}")
        print(f"  • URLs: {self.urls_file}")
        print(f"  • Statistics: {self.stats_file}")

def main():
    """Main entry point with flexible token configuration"""
    
    # Collect tokens from various sources
    tokens = []
    
    # 1. Check for multiple token environment variables
    for i in range(1, 21):  # Support up to 20 tokens
        token = os.getenv(f'GITHUB_TOKEN_{i}')
        if token:
            tokens.append(token)
    
    # 2. Check main token variable
    main_token = os.getenv('GITHUB_TOKEN')
    if main_token and main_token not in tokens:
        tokens.append(main_token)
    
    # 3. Check for tokens file
    if os.path.exists('tokens.txt'):
        with open('tokens.txt', 'r') as f:
            file_tokens = [line.strip() for line in f if line.strip()]
            tokens.extend([t for t in file_tokens if t not in tokens])
    
    # 4. Interactive input if no tokens found
    if not tokens:
        print("No GitHub tokens found automatically.")
        print("\nYou can provide tokens via:")
        print("  1. Environment variables: GITHUB_TOKEN, GITHUB_TOKEN_1, GITHUB_TOKEN_2, etc.")
        print("  2. A 'tokens.txt' file with one token per line")
        print("  3. Enter them now manually\n")
        
        num_tokens = input("How many tokens do you want to use? (or 'q' to quit): ")
        if num_tokens.lower() == 'q':
            sys.exit(0)
        
        try:
            num_tokens = int(num_tokens)
            for i in range(num_tokens):
                token = input(f"Enter token {i+1}: ").strip()
                if token:
                    tokens.append(token)
        except ValueError:
            print("Invalid input")
            sys.exit(1)
    
    if not tokens:
        print("❌ Error: At least one GitHub token is required.")
        print("\nCreate tokens at: https://github.com/settings/tokens")
        print("Required scope: public_repo")
        sys.exit(1)
    
    print(f"\n✅ Loaded {len(tokens)} GitHub tokens")
    
    # Start collection
    collector = MultiTokenCollector(tokens)
    
    try:
        collector.collect_parallel()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()