"""Lightweight performance evidence for the capstone submission."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd

from src.config.config import get_project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    args = parser.parse_args()
    root = get_project_root()
    path = root / "data/processed/customer_clv_segments.parquet"
    if not path.exists():
        raise SystemExit("Run the pipeline first: python scripts/run_pipeline.py")
    df = pd.read_parquet(path)
    sample = df.head(min(args.rows, len(df))).copy()
    start = time.perf_counter()
    _ = sample.sort_values("clv_expected_90d", ascending=False).head(100)
    elapsed = time.perf_counter() - start
    report = {
        "rows_benchmarked": int(len(sample)),
        "operation": "customer ranking by expected 90-day CLV",
        "elapsed_seconds": round(elapsed, 6),
        "rows_per_second": round(len(sample) / max(elapsed, 1e-9), 2),
    }
    out = root / "reports/performance_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
