#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "data/processed/customer_clv_segments.parquet" && ! -f "data/processed/customer_clv_segments.csv" ]]; then
  if [[ -f "data/raw/online_retail_II.xlsx" || -f "data/raw/online_retail_ii.zip" ]]; then
    echo "[Docker] Analytical artifacts missing; running the end-to-end pipeline."
    python scripts/run_pipeline.py
  else
    echo "[Docker] Raw UCI dataset not found. Starting the dashboard in setup mode."
    echo "[Docker] Place the UCI Online Retail II source under data/raw/ and run: python scripts/run_pipeline.py"
  fi
fi

exec python -m streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0
