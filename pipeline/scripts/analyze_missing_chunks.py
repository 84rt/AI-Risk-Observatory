import json
from pathlib import Path

# Load saved chunks
audit_dir = Path("data/results/comparison_audit")
with open(audit_dir / "fr_chunks_shell_2023.json", "r") as f:
    fr_chunks = json.load(f)
with open(audit_dir / "our_chunks_shell_2023.json", "r") as f:
    our_chunks = json.load(f)

print(f"FR Chunks: {len(fr_chunks)}")
print(f"Our Chunks: {len(our_chunks)}\n")

def find_overlap(target_chunk, other_chunks):
    # Normalize text for comparison (remove whitespace/newlines)
    target_text = "".join(target_chunk['chunk_text'].split()).lower()
    
    for i, other in enumerate(other_chunks):
        other_text = "".join(other['chunk_text'].split()).lower()
        if target_text[:100] in other_text or other_text[:100] in target_text:
            return i
    return None

print("--- Mapping Our Chunks to FR Chunks ---")
unmapped_our = []
for i, our_c in enumerate(our_chunks):
    match_idx = find_overlap(our_c, fr_chunks)
    if match_idx is not None:
        print(f"Our Chunk {i+1} matches FR Chunk {match_idx+1}")
    else:
        unmapped_our.append((i+1, our_c))

if unmapped_our:
    print(f"\n--- Unmapped Our Chunks ({len(unmapped_our)}) ---")
    for idx, chunk in unmapped_our:
        print(f"\n[OUR CHUNK {idx}]")
        print(f"Keywords: {chunk['matched_keywords']}")
        print(f"Text Snippet: {chunk['chunk_text'][:300]}...")
else:
    print("\nNo unmapped chunks found! (All our chunks were found in FR version)")
