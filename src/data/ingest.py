"""Data ingestion module for UCI Online Retail II dataset.

Downloads and extracts the official UCI dataset or loads it from local cache.
Official URL: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
"""

import os
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import requests
from tqdm import tqdm

from src.config.config import AppConfig, get_project_root, load_config


class DataIngestion:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.root = get_project_root()
        self.raw_dir = self.root / self.config.paths.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.zip_path = self.raw_dir / self.config.paths.raw_zip_name
        self.excel_path = self.raw_dir / self.config.paths.raw_excel_name
        self.cache_parquet = self.raw_dir / "raw_combined.parquet"
        self.cache_csv = self.raw_dir / "raw_combined.csv"

    def download_dataset(self, force: bool = False) -> bool:
        """Download official UCI Online Retail II zip archive if not already downloaded."""
        if not force and (self.excel_path.exists() or self.zip_path.exists()):
            print(f"[Ingest] Dataset already present in {self.raw_dir}")
            return True

        url = self.config.data_source.url
        print(f"[Ingest] Downloading official UCI Online Retail II dataset from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            temp_zip = self.zip_path.with_suffix(".tmp")
            with open(temp_zip, "wb") as f, tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc="Downloading UCI Dataset",
            ) as bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            temp_zip.rename(self.zip_path)
            print(f"[Ingest] Download complete: {self.zip_path}")
            return True
        except Exception as e:
            print(f"[Ingest] ERROR downloading dataset: {e}")
            print(
                "[Ingest] MANUAL ACTION REQUIRED if offline:\n"
                f"Please manually download 'online_retail_II.xlsx' from "
                f"https://archive.ics.uci.edu/dataset/502/online%2Bretail%2Bii\n"
                f"and place it at: {self.excel_path}"
            )
            return False

    def extract_dataset(self) -> bool:
        """Extract the Excel file from zip archive if not already extracted."""
        if self.excel_path.exists():
            return True

        if not self.zip_path.exists():
            print(f"[Ingest] Zip file {self.zip_path} not found.")
            return False

        print(f"[Ingest] Extracting {self.zip_path}...")
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
                # Find inner Excel file
                members = zip_ref.namelist()
                for member in members:
                    if member.lower().endswith((".xlsx", ".xls")):
                        source = zip_ref.open(member)
                        with open(self.excel_path, "wb") as target:
                            target.write(source.read())
                        print(f"[Ingest] Extracted {member} -> {self.excel_path}")
                        return True
            print("[Ingest] No Excel file found in archive.")
            return False
        except Exception as e:
            print(f"[Ingest] ERROR extracting archive: {e}")
            return False

    def load_raw_data(
        self,
        use_cache: bool = True,
        nrows: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load raw dataset from cache, Excel, or trigger download and extraction."""
        # Check cache first for faster development / test iterations
        if use_cache and nrows is None and self.cache_parquet.exists():
            print(f"[Ingest] Loading from raw cache: {self.cache_parquet}")
            return pd.read_parquet(self.cache_parquet)

        if not self.excel_path.exists():
            if not self.download_dataset():
                raise FileNotFoundError(
                    f"Dataset not available at {self.excel_path}. Please place "
                    f"'online_retail_II.xlsx' in {self.raw_dir}"
                )
            if not self.extract_dataset():
                raise FileNotFoundError(f"Failed to extract {self.excel_path}")

        print(f"[Ingest] Reading Excel sheets from {self.excel_path}...")
        dfs = []
        excel_file = pd.ExcelFile(self.excel_path)
        for sheet_name in self.config.data_source.expected_sheets:
            if sheet_name in excel_file.sheet_names:
                print(f"[Ingest] Loading sheet '{sheet_name}'...")
                df_sheet = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    nrows=nrows,
                )
                df_sheet["sheet_source"] = sheet_name
                dfs.append(df_sheet)
            else:
                print(f"[Ingest] Warning: Expected sheet '{sheet_name}' not found.")

        if not dfs:
            raise ValueError(f"No valid sheets found in {self.excel_path}")

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"[Ingest] Loaded combined raw data with {len(combined_df):,} records.")

        # Save to cache if full dataset loaded
        if nrows is None and use_cache:
            try:
                for col in ["Invoice", "StockCode", "Description", "Country"]:
                    if col in combined_df.columns:
                        combined_df[col] = combined_df[col].astype(str)
                combined_df.to_parquet(self.cache_parquet, index=False)
                print(f"[Ingest] Cached raw combined data to {self.cache_parquet}")
            except Exception as e:
                print(f"[Ingest] Note: Parquet cache skipped ({e}). Falling back to CSV.")
                combined_df.to_csv(self.cache_csv, index=False)

        return combined_df


def get_raw_data(use_cache: bool = True, nrows: Optional[int] = None) -> pd.DataFrame:
    """Convenience helper to retrieve raw dataset."""
    ingestion = DataIngestion()
    return ingestion.load_raw_data(use_cache=use_cache, nrows=nrows)


if __name__ == "__main__":
    ingestion = DataIngestion()
    if ingestion.download_dataset():
        ingestion.extract_dataset()
        df = ingestion.load_raw_data(nrows=100)
        print("\nRaw Data Sample:")
        print(df.head())
