#!/usr/bin/env python3
"""Add adoption_substantiveness, vendor_substantiveness, and risk_sub_substantiveness
columns to the mentions table.

This migration is idempotent: columns that already exist are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings  # noqa: E402

NEW_COLUMNS = [
    # (column_name, sql_type, comment)
    ("adoption_substantiveness", "TEXT", "boilerplate | moderate | substantive, from standalone classifier"),
    ("vendor_substantiveness",   "TEXT", "boilerplate | moderate | substantive, from standalone classifier"),
    ("risk_sub_substantiveness", "TEXT", "boilerplate | moderate | substantive, from standalone classifier (compare vs risk_substantiveness)"),
]


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(
        description="Add adoption/vendor/risk_sub substantiveness columns to mentions table."
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=settings.database_path,
        help=f"Path to SQLite DB (default: {settings.database_path})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned actions; do not modify DB.",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a .bak copy before migration.",
    )
    return p.parse_args()


def get_existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {row[1] for row in rows}


def main() -> None:
    args = parse_args()
    db_path = args.db_path

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        existing = get_existing_columns(conn, "mentions")

        to_add = [(col, typ, comment) for col, typ, comment in NEW_COLUMNS if col not in existing]
        already_present = [col for col, _, _ in NEW_COLUMNS if col in existing]

        if already_present:
            print(f"Already present (skipped): {', '.join(already_present)}")

        if not to_add:
            print("Nothing to do — all columns already exist.")
            return

        for col, typ, comment in to_add:
            print(f"  + {col}  {typ}  -- {comment}")

        if args.dry_run:
            print("Dry run — no changes made.")
            return

        if not args.no_backup:
            backup_path = db_path.with_suffix(db_path.suffix + ".bak")
            shutil.copy2(db_path, backup_path)
            print(f"Backup created: {backup_path}")

        with conn:
            for col, typ, _ in to_add:
                conn.execute(f'ALTER TABLE mentions ADD COLUMN "{col}" {typ}')
                print(f"Added column: {col}")

        print("Migration complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
