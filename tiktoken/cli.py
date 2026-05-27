#!/usr/bin/env python3
"""
Command-line interface for tiktoken.

This CLI tool allows you to count tokens in files and directories directly
from the command line, which is useful for:
- Estimating context window usage for codebases
- Quick token counting without writing Python code
- Batch processing multiple files
- Integration with shell scripts and CI/CD pipelines

Usage:
    tiktoken count file.txt
    tiktoken count --model gpt-4o file.txt
    tiktoken count --recursive ./src/
    tiktoken count --json file.txt
"""

import argparse
import sys
import json
import glob as glob_module
from pathlib import Path
from typing import List, Dict, Any, Optional

import tiktoken


def count_tokens_in_text(text: str, encoding_name: str) -> int:
    """Count tokens in a text string."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def count_tokens_in_file(file_path: Path, encoding_name: str) -> Optional[Dict[str, Any]]:
    """Count tokens in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        token_count = count_tokens_in_text(content, encoding_name)
        
        return {
            'file': str(file_path),
            'tokens': token_count,
            'chars': len(content),
            'lines': content.count('\n') + 1
        }
    except UnicodeDecodeError:
        return None  # Skip binary files
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return None


def get_encoding_for_model(model: str) -> str:
    """Get encoding name for a model."""
    try:
        enc = tiktoken.encoding_for_model(model)
        return enc.name
    except KeyError:
        print(f"Warning: Unknown model '{model}', using o200k_base encoding", file=sys.stderr)
        return "o200k_base"


def collect_files(paths: List[str], recursive: bool, pattern: Optional[str]) -> List[Path]:
    """Collect all files to process."""
    files = []
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            if recursive:
                if pattern:
                    files.extend(path.rglob(pattern))
                else:
                    files.extend(p for p in path.rglob('*') if p.is_file())
            else:
                if pattern:
                    files.extend(path.glob(pattern))
                else:
                    files.extend(p for p in path.glob('*') if p.is_file())
        else:
            # Try as glob pattern
            matched = list(Path('.').glob(path_str))
            files.extend(p for p in matched if p.is_file())
    
    return files


def format_output_text(results: List[Dict[str, Any]], summary: bool, per_file: bool) -> str:
    """Format output as plain text."""
    output = []
    
    if per_file and len(results) > 1:
        # Per-file breakdown
        for result in results:
            output.append(f"{result['file']}: {result['tokens']:,} tokens")
        output.append("")  # Blank line before summary
    
    if summary or len(results) > 1:
        # Summary statistics
        total_tokens = sum(r['tokens'] for r in results)
        total_chars = sum(r['chars'] for r in results)
        total_files = len(results)
        
        output.append(f"Total files: {total_files}")
        output.append(f"Total tokens: {total_tokens:,}")
        output.append(f"Total characters: {total_chars:,}")
        
        if total_files > 0:
            output.append(f"Average tokens per file: {total_tokens // total_files:,}")
    elif len(results) == 1:
        # Single file - just show token count
        output.append(f"{results[0]['tokens']:,}")
    
    return '\n'.join(output)


def format_output_json(results: List[Dict[str, Any]]) -> str:
    """Format output as JSON."""
    total_tokens = sum(r['tokens'] for r in results)
    total_chars = sum(r['chars'] for r in results)
    
    output = {
        'summary': {
            'total_files': len(results),
            'total_tokens': total_tokens,
            'total_characters': total_chars,
            'average_tokens_per_file': total_tokens // len(results) if results else 0
        },
        'files': results
    }
    
    return json.dumps(output, indent=2)


def format_output_csv(results: List[Dict[str, Any]]) -> str:
    """Format output as CSV."""
    lines = ["file,tokens,characters,lines"]
    for result in results:
        lines.append(f"{result['file']},{result['tokens']},{result['chars']},{result['lines']}")
    return '\n'.join(lines)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='tiktoken',
        description='Count tokens in files using tiktoken',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s count file.txt                    # Count tokens in a file
  %(prog)s count --model gpt-4o file.txt     # Use specific model encoding
  %(prog)s count -r ./src/                   # Count tokens in all files recursively
  %(prog)s count --glob "*.py" ./project/    # Count tokens in Python files
  %(prog)s count --json file.txt             # Output as JSON
  %(prog)s count --per-file ./src/           # Show per-file breakdown
"""
    )
    
    parser.add_argument(
        'command',
        choices=['count'],
        help='Command to execute (currently only "count" is supported)'
    )
    
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files or directories to count tokens in'
    )
    
    parser.add_argument(
        '-m', '--model',
        default=None,
        help='OpenAI model to use for encoding (e.g., gpt-4o, gpt-4-turbo)'
    )
    
    parser.add_argument(
        '-e', '--encoding',
        default='o200k_base',
        help='Encoding to use (default: o200k_base)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    
    parser.add_argument(
        '-g', '--glob',
        default=None,
        help='Glob pattern to filter files (e.g., "*.py")'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Output results as CSV'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Show summary statistics'
    )
    
    parser.add_argument(
        '--per-file',
        action='store_true',
        help='Show per-file token counts'
    )
    
    args = parser.parse_args()
    
    # Determine encoding to use
    if args.model:
        encoding_name = get_encoding_for_model(args.model)
    else:
        encoding_name = args.encoding
    
    # Collect files to process
    files = collect_files(args.paths, args.recursive, args.glob)
    
    if not files:
        print("No files found to process", file=sys.stderr)
        return 1
    
    # Process files
    results = []
    for file_path in files:
        result = count_tokens_in_file(file_path, encoding_name)
        if result:
            results.append(result)
    
    if not results:
        print("No files could be processed", file=sys.stderr)
        return 1
    
    # Format and output results
    if args.json:
        print(format_output_json(results))
    elif args.csv:
        print(format_output_csv(results))
    else:
        print(format_output_text(results, args.summary, args.per_file))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
