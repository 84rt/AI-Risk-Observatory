#!/usr/bin/env python3
"""Migrate mentions.risk_substantiveness from numeric schema to categorical TEXT.

This migration is idempotent and safe to re-run.
It rebuilds the `mentions` table when the column is not TEXT-like and normalizes
values to one of: boilerplate | moderate | substantive.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src.config import get_settings  # noqa: E402


VALID_LEVELS = {"boilerplate", "moderate", "substantive"}
ALIASES = {"contextual": "moderate"}


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    p = argparse.ArgumentParser(
        description=(
            "Migrate mentions.risk_substantiveness to TEXT and normalize values "
            "(boilerplate|moderate|substantive)."
        )
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


def normalize_substantiveness(value: object) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        score = float(value)
        if score <= 1.0:
            if score >= 0.67:
                return "substantive"
            if score >= 0.34:
                return "moderate"
            return "boilerplate"
        rounded = int(round(score))
        if rounded >= 3:
            return "substantive"
        if rounded == 2:
            return "moderate"
        return "boilerplate"

    token = str(value).strip().lower()
    token = ALIASES.get(token, token)
    if token in VALID_LEVELS:
        return token

    # Accept numeric strings.
    try:
        numeric = float(token)
    except ValueError:
        return None
    return normalize_substantiveness(numeric)


def get_column_type(conn: sqlite3.Connection, table: str, column: str) -> Optional[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    for _, name, col_type, *_ in rows:
        if name == column:
            return (col_type or "").upper().strip()
    return None


def is_text_type(col_type: Optional[str]) -> bool:
    if not col_type:
        return False
    return any(token in col_type for token in ("TEXT", "CHAR", "CLOB"))


def rebuild_mentions_with_text_column(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='mentions'"
    ).fetchone()
    if not row or not row[0]:
        raise RuntimeError("Could not read CREATE TABLE statement for 'mentions'.")
    create_sql = str(row[0])

    new_create = re.sub(
        r'CREATE TABLE\s+"?mentions"?',
        "CREATE TABLE mentions__new",
        create_sql,
        count=1,
        flags=re.IGNORECASE,
    )
    new_create, replaced = re.subn(
        r'(\brisk_substantiveness\b\s+)([A-Za-z0-9_()]+)',
        r"\1TEXT",
        new_create,
        count=1,
        flags=re.IGNORECASE,
    )
    if replaced != 1:
        raise RuntimeError("Could not patch risk_substantiveness type in CREATE TABLE SQL.")

    conn.execute(new_create)

    cols = [r[1] for r in conn.execute("PRAGMA table_info('mentions')").fetchall()]
    quoted = ", ".join(f'"{c}"' for c in cols)
    conn.execute(f'INSERT INTO mentions__new ({quoted}) SELECT {quoted} FROM mentions')
    conn.execute("DROP TABLE mentions")
    conn.execute("ALTER TABLE mentions__new RENAME TO mentions")


def normalize_values(conn: sqlite3.Connection) -> tuple[int, int]:
    rows = conn.execute(
        "SELECT rowid, risk_substantiveness FROM mentions"
    ).fetchall()
    updated = 0
    nullified = 0
    for rowid, value in rows:
        normalized = normalize_substantiveness(value)
        # Keep exactly canonical values or NULL.
        if normalized is None:
            if value is not None:
                conn.execute(
                    "UPDATE mentions SET risk_substantiveness = NULL WHERE rowid = ?",
                    (rowid,),
                )
                updated += 1
                nullified += 1
            continue

        if str(value) != normalized:
            conn.execute(
                "UPDATE mentions SET risk_substantiveness = ? WHERE rowid = ?",
                (normalized, rowid),
            )
            updated += 1
    return updated, nullified


def main() -> None:
    args = parse_args()
    db_path = args.db_path
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        col_type = get_column_type(conn, "mentions", "risk_substantiveness")
        if col_type is None:
            raise SystemExit("Column mentions.risk_substantiveness not found.")

        print(f"Current column type: {col_type or 'UNKNOWN'}")
        needs_rebuild = not is_text_type(col_type)

        if args.dry_run:
            print(f"Needs table rebuild: {'yes' if needs_rebuild else 'no'}")
            print("Would normalize values to: boilerplate | moderate | substantive | NULL")
            return

        if not args.no_backup:
            backup_path = db_path.with_suffix(db_path.suffix + ".bak")
            shutil.copy2(db_path, backup_path)
            print(f"Backup created: {backup_path}")

        with conn:
            if needs_rebuild:
                print("Rebuilding mentions table with TEXT risk_substantiveness...")
                rebuild_mentions_with_text_column(conn)

            updated, nullified = normalize_values(conn)
            print(f"Normalized rows updated: {updated}")
            print(f"Rows set to NULL (unrecognized legacy values): {nullified}")

        new_type = get_column_type(conn, "mentions", "risk_substantiveness")
        print(f"New column type: {new_type or 'UNKNOWN'}")
        if not is_text_type(new_type):
            raise RuntimeError("Migration completed but column is still not TEXT-like.")

        print("Migration complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

