#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DASH_DIR="$ROOT_DIR/dashboard"

SRC_ANNOTATIONS="$ROOT_DIR/data/golden_set/phase2_annotated/annotations.jsonl"
SRC_COMPANIES="$ROOT_DIR/data/reference/golden_set_companies.csv"

DEST_DIR="$DASH_DIR/data"
DEST_ANNOTATIONS="$DEST_DIR/annotations.jsonl"
DEST_COMPANIES="$DEST_DIR/golden_set_companies.csv"

if [[ ! -f "$SRC_ANNOTATIONS" ]]; then
  echo "Missing source annotations file: $SRC_ANNOTATIONS" >&2
  exit 1
fi

if [[ ! -f "$SRC_COMPANIES" ]]; then
  echo "Missing source companies file: $SRC_COMPANIES" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC_ANNOTATIONS" "$DEST_ANNOTATIONS"
cp "$SRC_COMPANIES" "$DEST_COMPANIES"

echo "Dashboard data synced to: $DEST_DIR"
ls -lh "$DEST_ANNOTATIONS" "$DEST_COMPANIES"
