#!/usr/bin/env bash
# refresh_data.sh — keep the AIRO data current
#
# Run this script regularly (e.g. weekly via cron or manually) to:
#   1. Pull new filings from FinancialReports.eu (FR API)
#   2. Rebuild target_manifest.csv
#
# Usage:
#   ./scripts/refresh_data.sh              # standard refresh (FR only)
#   ./scripts/refresh_data.sh --full       # also re-fetch CH period-of-accounts
#
# After this runs, feed newly available markdown through the annotation
# pipeline (run_full_100_report_pipeline.py or equivalent) so the dashboard
# picks up fresh data automatically via annotations.jsonl.

set -euo pipefail
cd "$(dirname "$0")/.."

FULL=false
for arg in "$@"; do
  [[ "$arg" == "--full" ]] && FULL=true
done

echo "=== AIRO data refresh — $(date '+%Y-%m-%d %H:%M') ==="

# ── Step 1 (optional): re-fetch Companies House period-of-accounts ────────────
# Needed when new companies join the LSE or CH filing history changes.
# Slow (~30 min for 1,400 companies); skip unless doing a full refresh.
if [[ "$FULL" == "true" ]]; then
  echo ""
  echo "── Step 1: Fetching CH period-of-accounts (--full mode) ─────────────────"
  python3 scripts/fetch_ch_period_of_accounts.py
fi

# ── Step 2: Pull new FR filings & markdown ────────────────────────────────────
echo ""
echo "── Step 2: Refreshing FR status & fetching new markdown ─────────────────"
python3 scripts/refresh_fr_status.py --check-not-in-fr --check-year-gaps --rebuild-manifest

echo ""
echo "── Done ─────────────────────────────────────────────────────────────────"
echo "   Manifest: data/reference/target_manifest.csv"
echo "   Next: run annotation pipeline on newly fetched markdown files"
echo "         then the dashboard will pick up new data automatically."
