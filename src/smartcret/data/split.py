"""Data splitting utilities ensuring no data leakage and preserving source integrity."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test_reserved: pd.DataFrame

    def summary(self) -> str:
        return (
            f"Dataset Splits:\n"
            f"  - Train: {len(self.train)} filas\n"
            f"  - Validation: {len(self.val)} filas\n"
            f"  - Test reservado: {len(self.test_reserved)} filas"
        )


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_by_source: bool = True,
) -> DatasetSplits:
    """Splits dataframe into Train, Validation, and Reserved Test sets.

    Guarantees no data leakage and fixed seed reproducibility.
    """
    assert np.isclose(
        train_size + val_size + test_size, 1.0
    ), "Las proporciones deben sumar 1.0"

    strat = df["fuente"] if (stratify_by_source and "fuente" in df.columns and df["fuente"].nunique() > 1) else None

    n_total = len(df)
    n_test = int(round(n_total * test_size))
    n_val = int(round(n_total * val_size))

    # Step 1: Separate Reserved Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=n_test,
        random_state=random_state,
        stratify=strat,
    )

    # Step 2: Separate Train and Validation
    strat_train = (
        train_val_df["fuente"]
        if (stratify_by_source and "fuente" in train_val_df.columns and train_val_df["fuente"].nunique() > 1)
        else None
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=n_val,
        random_state=random_state,
        stratify=strat_train,
    )

    test_df = test_df.copy()
    train_df = train_df.copy()
    val_df = val_df.copy()

    return DatasetSplits(
        train=train_df,
        val=val_df,
        test_reserved=test_df,
    )


def save_splits(
    splits: DatasetSplits, output_dir: Union[str, Path] = "data/processed"
) -> None:
    """Saves splits to CSV files in output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(out / "train.csv", index=False)
    splits.val.to_csv(out / "val.csv", index=False)
    splits.test_reserved.to_csv(out / "test_reserved.csv", index=False)
