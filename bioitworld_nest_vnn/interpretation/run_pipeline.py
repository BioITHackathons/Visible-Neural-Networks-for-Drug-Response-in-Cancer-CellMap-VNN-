"""CLI entry point for the NeST-VNN LLM interpretation pipeline.

Usage:
    python interpretation/run_pipeline.py breast_msk_2025 binary_os_months
    python interpretation/run_pipeline.py breast_msk_2025 binary_os_months --batch-size 5 --limit 10
    python interpretation/run_pipeline.py breast_msk_2025 binary_os_months --no-skip
"""

import argparse
import sys
from pathlib import Path

# Allow running as a top-level script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from interpretation.pipeline import run_pipeline
from interpretation.db import DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Run LLM interpretation pipeline for NeST-VNN")
    parser.add_argument("study_id", help="Study identifier (e.g. breast_msk_2025)")
    parser.add_argument("label", help="Endpoint label (e.g. binary_os_months)")
    parser.add_argument("--batch-size", type=int, default=5, help="Patients per LLM call (default 5)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N patients")
    parser.add_argument("--no-skip", action="store_true", help="Re-process already stored patients")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    args = parser.parse_args()

    n_ok, n_err = run_pipeline(
        study_id=args.study_id,
        label=args.label,
        batch_size=args.batch_size,
        limit=args.limit,
        skip_existing=not args.no_skip,
        db_path=args.db,
    )

    sys.exit(0 if n_err == 0 else 1)


if __name__ == "__main__":
    main()
