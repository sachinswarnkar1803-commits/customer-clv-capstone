"""Temporal train-validation-holdout split module.

Guarantees zero future leakage by partitioning transactions strictly by timestamp.
"""

from typing import Optional, Tuple
import pandas as pd
from src.config.config import AppConfig, load_config


class TemporalSplitter:
    """Partitions transactional datasets across time horizons to prevent leakage."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()

    def split_train_holdout(
        self,
        df: pd.DataFrame,
        cutoff_date: Optional[str] = None,
        date_col: str = "transaction_date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split transactions into an observation/training window and holdout window.

        Args:
            df: Clean transactions DataFrame.
            cutoff_date: Cutoff timestamp string (e.g. '2010-12-09'). If None, loaded from config.
            date_col: Date column name.

        Returns:
            train_df: Transactions with date <= cutoff_date.
            holdout_df: Transactions with date > cutoff_date.
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])

        cutoff = pd.to_datetime(cutoff_date or self.config.temporal_splits.train_end_date)

        train_df = df[df[date_col] <= cutoff].copy()
        holdout_df = df[df[date_col] > cutoff].copy()

        print(f"[TemporalSplitter] Cutoff Date: {cutoff.strftime('%Y-%m-%d')}")
        print(f"[TemporalSplitter] Training Records: {len(train_df):,} | "
              f"Date Span: {train_df[date_col].min().strftime('%Y-%m-%d')} to {train_df[date_col].max().strftime('%Y-%m-%d')}")
        print(f"[TemporalSplitter] Holdout Records:  {len(holdout_df):,} | "
              f"Date Span: {holdout_df[date_col].min().strftime('%Y-%m-%d')} to {holdout_df[date_col].max().strftime('%Y-%m-%d')}")

        # Strict leakage assertion
        assert train_df[date_col].max() <= cutoff, "CRITICAL: Training data contains records beyond cutoff!"
        if len(holdout_df) > 0:
            assert holdout_df[date_col].min() > cutoff, "CRITICAL: Holdout data contains records before cutoff!"

        return train_df, holdout_df
