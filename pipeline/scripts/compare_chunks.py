#!/usr/bin/env python3
"""
Compare two JSONL chunk files side by side with color-coded differences.

USAGE:
    python compare_chunks.py <original.jsonl> <processed.jsonl> [OPTIONS]

OPTIONS:
    --all         Show all chunks (default: only show differences)
    --limit N     Show only first N chunks with differences
    --truncate    Limit display to 500 chars per chunk

OUTPUT:
    🔴 Red = removed/changed text from original
    🟢 Green = new/corrected text in processed version

EXAMPLES:
    python compare_chunks.py chunks.jsonl chunks_gemma.jsonl
    python compare_chunks.py chunks.jsonl chunks_gemma.jsonl --limit 5
    python compare_chunks.py chunks.jsonl chunks_gemma.jsonl --truncate
"""

import json
import sys
from pathlib import Path
from difflib import SequenceMatcher
from typing import List, Tuple


def load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file into a list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def highlight_diff(text1: str, text2: str) -> Tuple[str, str]:
    """
    Highlight differences between two texts using ANSI color codes.
    Returns tuple of (highlighted_text1, highlighted_text2)
    """
    # ANSI color codes
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    
    matcher = SequenceMatcher(None, text1, text2)
    result1 = []
    result2 = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            result1.append(text1[i1:i2])
            result2.append(text2[j1:j2])
        elif tag == 'replace':
            result1.append(f"{RED}{text1[i1:i2]}{RESET}")
            result2.append(f"{GREEN}{text2[j1:j2]}{RESET}")
        elif tag == 'delete':
            result1.append(f"{RED}{text1[i1:i2]}{RESET}")
        elif tag == 'insert':
            result2.append(f"{GREEN}{text2[j1:j2]}{RESET}")
    
    return ''.join(result1), ''.join(result2)


def wrap_text(text: str, width: int = 80) -> List[str]:
    """Simple text wrapping that preserves ANSI codes."""
    lines = []
    current_line = ""
    current_visible_length = 0
    in_ansi = False
    
    for char in text:
        if char == '\033':
            in_ansi = True
        
        current_line += char
        
        if not in_ansi:
            if char == '\n':
                lines.append(current_line)
                current_line = ""
                current_visible_length = 0
            else:
                current_visible_length += 1
                if current_visible_length >= width:
                    lines.append(current_line)
                    current_line = ""
                    current_visible_length = 0
        
        if in_ansi and char == 'm':
            in_ansi = False
    
    if current_line:
        lines.append(current_line)
    
    return lines


def print_divider(width: int = 160):
    """Print a dividing line."""
    print("=" * width)


def print_chunk_comparison(chunk1: dict, chunk2: dict, index: int, truncate: bool = False):
    """Print a side-by-side comparison of two chunks."""
    chunk_id = chunk1['chunk_id']
    text1 = chunk1['chunk_text']
    text2 = chunk2['chunk_text']
    
    # Check if texts are different
    if text1 == text2:
        return False  # No difference
    
    print_divider()
    print(f"\n📄 Chunk #{index + 1}: {chunk_id}")
    print(f"   Company: {chunk1['company_name']} ({chunk1['report_year']})")
    print(f"   Document: {chunk1['document_id']}")
    print()
    
    # Highlight differences
    highlighted1, highlighted2 = highlight_diff(text1, text2)
    
    # Print side by side
    print("🔴 ORIGINAL (chunks.jsonl):")
    print("-" * 80)
    if truncate and len(highlighted1) > 500:
        print(highlighted1[:500] + "...")
    else:
        print(highlighted1)
    print()
    
    print("🟢 GEMMA-PROCESSED (chunks_gemma.jsonl):")
    print("-" * 80)
    if truncate and len(highlighted2) > 500:
        print(highlighted2[:500] + "...")
    else:
        print(highlighted2)
    print()
    
    return True  # Had differences


def compare_files(file1: Path, file2: Path, show_all: bool = False, max_chunks: int = None, truncate: bool = False):
    """Compare two JSONL files."""
    print(f"\n🔍 Comparing chunks files:")
    print(f"   Original: {file1.name}")
    print(f"   Gemma:    {file2.name}")
    print()
    
    chunks1 = load_jsonl(file1)
    chunks2 = load_jsonl(file2)
    
    print(f"✅ Loaded {len(chunks1)} chunks from original file")
    print(f"✅ Loaded {len(chunks2)} chunks from Gemma file")
    
    if len(chunks1) != len(chunks2):
        print(f"\n⚠️  WARNING: File lengths don't match!")
        return
    
    differences_found = 0
    chunks_compared = 0
    
    for i, (c1, c2) in enumerate(zip(chunks1, chunks2)):
        if max_chunks and chunks_compared >= max_chunks:
            break
            
        if c1['chunk_id'] != c2['chunk_id']:
            print(f"\n⚠️  Chunk ID mismatch at position {i}!")
            continue
        
        chunks_compared += 1
        
        if show_all or c1['chunk_text'] != c2['chunk_text']:
            if print_chunk_comparison(c1, c2, i, truncate):
                differences_found += 1
    
    print_divider()
    print(f"\n📊 Summary:")
    print(f"   Total chunks compared: {chunks_compared}")
    print(f"   Chunks with differences: {differences_found}")
    print(f"   Chunks identical: {chunks_compared - differences_found}")
    print()


def main():
    """Main entry point."""
    # Parse arguments
    if len(sys.argv) < 3:
        print("Usage: python compare_chunks.py <original.jsonl> <gemma.jsonl> [--all] [--limit N] [--truncate]")
        print("\nOptions:")
        print("  --all        Show all chunks, even identical ones")
        print("  --limit N    Only compare first N chunks with differences")
        print("  --truncate   Truncate long chunks to 500 characters (default: show full chunks)")
        sys.exit(1)
    
    file1 = Path(sys.argv[1])
    file2 = Path(sys.argv[2])
    
    show_all = '--all' in sys.argv
    truncate = '--truncate' in sys.argv
    max_chunks = None
    
    if '--limit' in sys.argv:
        try:
            limit_idx = sys.argv.index('--limit')
            max_chunks = int(sys.argv[limit_idx + 1])
        except (ValueError, IndexError):
            print("Error: --limit requires a number")
            sys.exit(1)
    
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)
    
    compare_files(file1, file2, show_all, max_chunks, truncate)


if __name__ == '__main__':
    main()
