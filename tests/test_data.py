"""Unit tests for Phase 1: Data validation, generation, and splitting."""

import pytest
import pandas as pd
from smartcret.data.loader import DataLoader
from smartcret.data.generate_synthetic import generate_synthetic_data
from smartcret.data.split import split_dataset


def test_synthetic_data_generation():
    df = generate_synthetic_data(n_samples=500, random_seed=42)
    assert len(df) == 500
    assert "cemento_kg" in df.columns
    assert "resistencia_mpa" in df.columns
    assert "slump_mm" in df.columns
    assert (df["fuente"] == "sintetico").all()
    assert (df["resistencia_mpa"] >= 10.0).all()
    assert (df["slump_mm"] >= 0.0).all()


def test_data_loader_validation():
    loader = DataLoader("configs/schema.yaml")
    df = generate_synthetic_data(n_samples=100, random_seed=123)
    clean_df, report = loader.validate_and_clean(df)

    assert report.is_valid
    assert len(clean_df) == 100
    assert report.invalid_rows == 0
    assert "relacion_ac" in clean_df.columns


def test_data_loader_catches_invalid_values():
    loader = DataLoader("configs/schema.yaml")
    df = generate_synthetic_data(n_samples=10, random_seed=999)
    # Inject out-of-range cement value
    df.loc[0, "cemento_kg"] = 50.0  # min is 100.0
    clean_df, report = loader.validate_and_clean(df, drop_invalid=True)

    assert len(clean_df) == 9
    assert report.invalid_rows == 1


def test_split_dataset_no_leakage():
    df = generate_synthetic_data(n_samples=200, random_seed=42)
    splits = split_dataset(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)

    assert len(splits.train) + len(splits.val) + len(splits.test_reserved) == 200
    assert len(splits.train) == 140
    assert len(splits.val) == 30
    assert len(splits.test_reserved) == 30

    # Ensure no index overlap
    assert set(splits.train.index).isdisjoint(splits.test_reserved.index)
