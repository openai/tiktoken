# tiktoken CLI

Command-line interface for counting tokens in files and directories.

## Installation

After installing tiktoken, the `tiktoken` command will be available:

```bash
pip install tiktoken
```

## Usage

### Basic Token Counting

Count tokens in a single file:
```bash
tiktoken count file.txt
```

Output:
```
42
```

### Using Specific Models

Count tokens using a specific model's encoding:
```bash
tiktoken count --model gpt-4o document.txt
tiktoken count --model gpt-4-turbo code.py
```

### Directory Operations

Count tokens in all files in a directory:
```bash
tiktoken count --recursive ./src/
```

Use glob patterns to filter files:
```bash
tiktoken count --glob "*.py" ./project/
tiktoken count --recursive --glob "*.md" ./docs/
```

### Output Formats

#### JSON Output
```bash
tiktoken count --json file.txt
```

Output:
```json
{
  "summary": {
    "total_files": 1,
    "total_tokens": 1250,
    "total_characters": 5432,
    "average_tokens_per_file": 1250
  },
  "files": [
    {
      "file": "file.txt",
      "tokens": 1250,
      "chars": 5432,
      "lines": 85
    }
  ]
}
```

#### CSV Output
```bash
tiktoken count --csv ./src/
```

Output:
```csv
file,tokens,characters,lines
src/main.py,450,2100,65
src/utils.py,320,1540,48
src/config.py,180,850,28
```

#### Per-File Breakdown
```bash
tiktoken count --per-file ./src/
```

Output:
```
src/main.py: 450 tokens
src/utils.py: 320 tokens
src/config.py: 180 tokens

Total files: 3
Total tokens: 950
Total characters: 4490
Average tokens per file: 316
```

## Use Cases

### Estimating Context Window Usage

Check if your codebase fits in a model's context window:

```bash
# GPT-4 Turbo has 128k token context
tiktoken count --model gpt-4-turbo --recursive ./my-project/

# Output: Total tokens: 45,230
# Result: Fits comfortably in context window
```

### Cost Estimation

Estimate API costs by counting tokens:

```bash
tiktoken count --json --recursive ./documents/ > token_report.json
# Use the token count to calculate costs based on model pricing
```

### CI/CD Integration

Add token counting to your CI pipeline:

```bash
#!/bin/bash
TOKEN_COUNT=$(tiktoken count --recursive ./src/ | grep "Total tokens" | awk '{print $3}' | tr -d ',')
MAX_TOKENS=50000

if [ $TOKEN_COUNT -gt $MAX_TOKENS ]; then
  echo "Error: Codebase exceeds $MAX_TOKENS tokens (found: $TOKEN_COUNT)"
  exit 1
fi
```

### Documentation Analysis

Analyze documentation token usage:

```bash
tiktoken count --recursive --glob "*.md" --per-file ./docs/ | tee docs_tokens.txt
```

## Command Reference

### Arguments

- `paths`: One or more files or directories to process

### Options

- `-m, --model MODEL`: Use encoding for specific OpenAI model (e.g., `gpt-4o`, `gpt-4-turbo`)
- `-e, --encoding ENCODING`: Specify encoding directly (default: `o200k_base`)
- `-r, --recursive`: Process directories recursively
- `-g, --glob PATTERN`: Filter files using glob pattern (e.g., `"*.py"`)
- `--json`: Output results as JSON
- `--csv`: Output results as CSV
- `--summary`: Show summary statistics
- `--per-file`: Show per-file token counts

## Examples

### Count tokens in Python files
```bash
tiktoken count --glob "*.py" --recursive ./project/
```

### Generate JSON report for multiple files
```bash
tiktoken count --json file1.txt file2.txt file3.txt > report.json
```

### Check specific model compatibility
```bash
tiktoken count --model gpt-4o --summary ./codebase/
```

### Export to CSV for analysis
```bash
tiktoken count --csv --recursive ./src/ > tokens.csv
```

## Tips

1. **Performance**: The CLI processes files quickly thanks to tiktoken's fast Rust implementation
2. **Binary Files**: Binary files are automatically skipped
3. **Large Directories**: Use `--glob` to filter files and speed up processing
4. **Shell Integration**: Pipe output to other tools for further processing

## Troubleshooting

**Error: "No files found to process"**
- Check your glob pattern syntax
- Ensure files exist in the specified path
- Use `--recursive` for subdirectories

**Error: "Unknown model 'xyz'"**
- The model name might be incorrect
- Use `--encoding` instead to specify encoding directly
- Check [OpenAI's model documentation](https://platform.openai.com/docs/models) for valid model names

**Binary file warnings**
- The CLI automatically skips binary files
- This is expected behavior and can be ignored
