"""
Test suite for tiktoken CLI.

Run with: pytest tests/test_cli.py
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tiktoken.cli import (
    count_tokens_in_text,
    count_tokens_in_file,
    collect_files,
    format_output_json,
    format_output_csv,
)


def test_count_tokens_in_text():
    """Test basic token counting."""
    text = "Hello, world!"
    count = count_tokens_in_text(text, "o200k_base")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_in_file():
    """Test counting tokens in a file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a test file for tiktoken CLI.")
        temp_path = f.name
    
    try:
        result = count_tokens_in_file(Path(temp_path), "o200k_base")
        assert result is not None
        assert 'tokens' in result
        assert 'chars' in result
        assert 'lines' in result
        assert result['tokens'] > 0
    finally:
        os.unlink(temp_path)


def test_collect_files_single_file():
    """Test collecting a single file."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_path = f.name
    
    try:
        files = collect_files([temp_path], False, None)
        assert len(files) == 1
        assert files[0] == Path(temp_path)
    finally:
        os.unlink(temp_path)


def test_collect_files_directory():
    """Test collecting files from a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_dir = Path(tmpdir)
        (test_dir / "file1.txt").write_text("content 1")
        (test_dir / "file2.txt").write_text("content 2")
        
        files = collect_files([tmpdir], False, None)
        assert len(files) == 2


def test_format_output_json():
    """Test JSON output formatting."""
    results = [
        {'file': 'test.txt', 'tokens': 100, 'chars': 500, 'lines': 10}
    ]
    
    output = format_output_json(results)
    assert 'summary' in output
    assert 'total_tokens' in output
    assert '100' in output


def test_format_output_csv():
    """Test CSV output formatting."""
    results = [
        {'file': 'test.txt', 'tokens': 100, 'chars': 500, 'lines': 10}
    ]
    
    output = format_output_csv(results)
    assert 'file,tokens,characters,lines' in output
    assert 'test.txt,100,500,10' in output


if __name__ == '__main__':
    # Run basic tests
    print("Running tiktoken CLI tests...")
    
    test_count_tokens_in_text()
    print("✓ test_count_tokens_in_text")
    
    test_count_tokens_in_file()
    print("✓ test_count_tokens_in_file")
    
    test_collect_files_single_file()
    print("✓ test_collect_files_single_file")
    
    test_collect_files_directory()
    print("✓ test_collect_files_directory")
    
    test_format_output_json()
    print("✓ test_format_output_json")
    
    test_format_output_csv()
    print("✓ test_format_output_csv")
    
    print("\n✅ All tests passed!")
