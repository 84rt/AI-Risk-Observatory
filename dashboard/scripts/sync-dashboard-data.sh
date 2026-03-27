#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DASH_DIR="$ROOT_DIR/dashboard"

SRC_ANNOTATIONS="${SRC_ANNOTATIONS:-$ROOT_DIR/data/golden_set/phase2_annotated/annotations.jsonl}"
SRC_COMPANIES_NEW="${SRC_COMPANIES_NEW:-$ROOT_DIR/data/reference/companies_metadata_v2.csv}"
SRC_COMPANIES_OLD="${SRC_COMPANIES_OLD:-$ROOT_DIR/data/reference/golden_set_companies.csv}"
SRC_DOCUMENT_MONTHS="${SRC_DOCUMENT_MONTHS:-$ROOT_DIR/dashboard/data/document_months.json}"

DEST_DIR="$DASH_DIR/data"
DEST_ANNOTATIONS="$DEST_DIR/annotations.jsonl"
DEST_COMPANIES="$DEST_DIR/golden_set_companies.csv"
DEST_DOCUMENT_MONTHS="$DEST_DIR/document_months.json"

if [[ ! -f "$SRC_ANNOTATIONS" ]]; then
  echo "Missing source annotations file: $SRC_ANNOTATIONS" >&2
  exit 1
fi

if [[ -f "$SRC_COMPANIES_NEW" ]]; then
  SRC_COMPANIES="$SRC_COMPANIES_NEW"
else
  SRC_COMPANIES="$SRC_COMPANIES_OLD"
fi

if [[ ! -f "$SRC_COMPANIES" ]]; then
  echo "Missing source companies file: $SRC_COMPANIES_NEW (fallback: $SRC_COMPANIES_OLD)" >&2
  exit 1
fi

if [[ ! -f "$SRC_DOCUMENT_MONTHS" ]]; then
  echo "Missing source document months file: $SRC_DOCUMENT_MONTHS" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC_ANNOTATIONS" "$DEST_ANNOTATIONS"
cp "$SRC_COMPANIES" "$DEST_COMPANIES"
cp "$SRC_DOCUMENT_MONTHS" "$DEST_DOCUMENT_MONTHS"

echo "Dashboard data synced to: $DEST_DIR"
echo "Companies source: $SRC_COMPANIES"
echo "Document months source: $SRC_DOCUMENT_MONTHS"
ls -lh "$DEST_ANNOTATIONS" "$DEST_COMPANIES" "$DEST_DOCUMENT_MONTHS"
