"""Convenience runner script for the BDS-34 end-to-end capstone pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/default_config.yaml
    python scripts/run_pipeline.py --validate-only
    python scripts/run_pipeline.py --help
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="BDS-34 Probabilistic CLV Capstone Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                      # Full end-to-end pipeline
  python scripts/run_pipeline.py --validate-only      # Data ingestion + validation only
  python scripts/run_pipeline.py --config configs/custom.yaml
        """,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file (default: configs/default_config.yaml)")
    parser.add_argument("--validate-only", action="store_true", help="Run only data ingestion and validation, then exit.")
    args = parser.parse_args()

    if args.validate_only:
        print("[Runner] Running validation-only mode...")
        from src.validation.validator import run_validation_pipeline
        clean_df, audit = run_validation_pipeline(use_cache=True)
        print(f"\n[Runner] Validation complete: {len(clean_df):,} clean records. "
              f"Revenue: £{audit['metrics']['total_clean_revenue_gbp']:,.2f}")
        return

    from src.pipeline import run_full_pipeline
    success = run_full_pipeline(config_path=args.config)
    if not success:
        print("[Runner] Pipeline reported failure. Check logs.")
        sys.exit(1)
    print("[Runner] Pipeline completed. Launch dashboard: python -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
