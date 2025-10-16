# Multi-Token GitHub Collector

High-performance GitHub repository collector that uses multiple tokens in parallel to efficiently search for AI/LLM-related configuration files across GitHub.

## Features

- **Multi-token parallel processing** - Distribute searches across multiple GitHub tokens for maximum speed
- **Intelligent token management** - Automatically selects tokens with remaining rate limits
- **Rate limit handling** - Waits and auto-retries when rate limits are hit
- **Resumable state** - Save progress and resume interrupted collections
- **Comprehensive search** - Searches for Claude, Agent, and other AI configuration files
- **Detailed statistics** - Performance metrics, language distribution, and more

## Performance

| Tokens | Estimated Time | Repos/hour | Total Repos |
|--------|---------------|------------|-------------|
| 1      | ~25 hours     | 600        | 15,000      |
| 3      | ~8 hours      | 3,125      | 25,000      |
| **5**  | **~5 hours**  | **7,500**  | **37,500**  |
| 10     | ~2.5 hours    | 20,000     | 50,000      |

**Recommended: 5-10 tokens** for optimal balance between speed and setup complexity.

## Requirements

- Python 3.7+
- GitHub Personal Access Tokens (1 or more)

## Installation

1. **Clone or download this project**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure GitHub tokens** (see Configuration section below)

## Configuration

The collector supports multiple methods for providing GitHub tokens:

### Method 1: Environment Variables (Recommended)

```bash
# Set multiple tokens
export GITHUB_TOKEN_1="ghp_your_first_token_here"
export GITHUB_TOKEN_2="ghp_your_second_token_here"
export GITHUB_TOKEN_3="ghp_your_third_token_here"
export GITHUB_TOKEN_4="ghp_your_fourth_token_here"
export GITHUB_TOKEN_5="ghp_your_fifth_token_here"

# Or use the main token variable
export GITHUB_TOKEN="ghp_your_token_here"
```

You can use the included setup script:
```bash
./setup_tokens.sh
```

### Method 2: tokens.txt File

Create a `tokens.txt` file with one token per line:
```
ghp_your_first_token_here
ghp_your_second_token_here
ghp_your_third_token_here
ghp_your_fourth_token_here
ghp_your_fifth_token_here
```

### Method 3: Interactive Input

Run the script without tokens configured, and it will prompt you to enter them.

### Creating GitHub Tokens

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "Multi-Token Collector 1")
4. Select scope: **`public_repo`** (for public repositories)
5. Click "Generate token"
6. Copy the token immediately (you won't see it again)
7. Repeat for each additional token you want to use

**Note:** You can use multiple GitHub accounts or create multiple tokens from the same account.

## Usage

### Basic Usage

```bash
python multi_token_collector.py
```

### What It Searches For

The collector searches for these AI/LLM configuration files:
- **Claude/Anthropic**: `Claude.md`, `CLAUDE.md`, `.claude`
- **Agents**: `Agent.md`, `AGENT.md`, `agent.json`, `agent.yaml`
- **OpenAI/ChatGPT**: `openai.md`, `gpt.json`, `.chatgpt`
- **Cursor IDE**: `.cursorrules`, `cursor.json`
- **GitHub Copilot**: `.copilot`, `copilot.json`
- **AI Configuration**: `ai.json`, `llm.json`, `prompts.json`
- **LangChain**: `langchain.json`, `chains.yaml`
- And many more...

### Customizing Target Files

Edit the `target_files` list in `multi_token_collector.py` (around line 50) to search for different files.

## Output Files

The collector generates several output files in the `outputs/` directory:

| File | Description |
|------|-------------|
| `outputs/multi_token_results.json` | Complete repository data with metadata (stars, language, files found) |
| `outputs/multi_token_urls.txt` | Simple list of repository URLs (one per line) |
| `outputs/collection_statistics.json` | Detailed statistics and performance metrics |
| `outputs/multi_token_state.pkl` | Resumable state (for interrupted collections) |

### Example Output Structure

**multi_token_results.json:**
```json
{
  "timestamp": "2025-10-16T20:48:00",
  "total_repositories": 2847,
  "tokens_used": 5,
  "files_searched": 4,
  "repositories": [
    {
      "name": "example/repo",
      "url": "https://github.com/example/repo",
      "files_found": ["Claude.md", "Agent.md"],
      "stars": 1234,
      "language": "Python",
      "description": "AI-powered tool...",
      "collected_at": "2025-10-16T20:45:00"
    }
  ]
}
```

**collection_statistics.json:**
```json
{
  "total_repositories": 2847,
  "repos_with_multiple_files": 142,
  "token_performance": {
    "token_1": {"repos_collected": 587, "errors": 0},
    "token_2": {"repos_collected": 612, "errors": 0}
  },
  "language_distribution": {
    "Python": 1234,
    "JavaScript": 892
  },
  "file_distribution": {
    "Claude.md": 1456,
    "Agent.md": 982
  }
}
```

## How It Works

1. **Token Pool Management**: Maintains a pool of GitHub tokens with rate limit tracking
2. **Work Queue**: Files to search are queued for processing
3. **Worker Threads**: Each token runs in its own thread, processing files from the queue
4. **Dynamic Token Selection**: Automatically selects the token with the most remaining rate limit
5. **Rate Limit Handling**: When all tokens are exhausted, waits for the next reset time
6. **Result Aggregation**: Merges results, tracking which files are found in each repository
7. **Periodic Saving**: Saves progress every minute for resumability

### Rate Limits

GitHub API limits per token:
- **Search API**: 30 requests per hour
- **Core API**: 5,000 requests per hour

With 5 tokens, you get:
- 150 search requests per hour
- Can complete ~75 file searches in ~5 hours

## Resuming Interrupted Collections

If the collection is interrupted (Ctrl+C or crash), the state is automatically saved in `outputs/multi_token_state.pkl`. Simply run the script again to resume from where it left off.

To start fresh:
```bash
rm outputs/multi_token_state.pkl
# Or delete the entire outputs directory
rm -rf outputs/
python multi_token_collector.py
```

## Monitoring Progress

The script displays real-time progress:
```
📊 Progress: 45/75 files
⏱️  Rate: 15.2 files/min | ETA: 2.0 minutes
🗂️  Unique repos collected: 2847
🔑 Active tokens: 5/5
```

## Troubleshooting

### "No tokens available"
- Check your token configuration
- Verify tokens are valid at https://github.com/settings/tokens
- Ensure tokens have `public_repo` scope

### "Rate limit hit"
- Normal behavior - the script will wait and retry automatically
- Consider adding more tokens for faster collection

### "403 Forbidden"
- Token may be invalid or expired
- Token may lack required permissions
- GitHub may have detected unusual activity

## License

This tool is for educational and research purposes. Please respect GitHub's Terms of Service and API usage guidelines.

## Tips for Best Results

1. **Use 5-10 tokens** for optimal performance
2. **Run during off-peak hours** to avoid potential rate limit issues
3. **Check output periodically** to ensure collection is progressing
4. **Keep tokens secure** - never commit them to version control
5. **Use different GitHub accounts** for tokens if possible (more reliable than multiple tokens from same account)

## Performance Notes

- Each file search can return up to 1,000 results (GitHub limitation)
- Searches are paginated at 100 results per page
- 1-second delay between pages to be respectful to GitHub's API
- Typical collection with 5 tokens and 75 files: **4-6 hours**