"""Unit tests for Phase 2: Model training, predictor, and uncertainty estimation."""

import pytest
import numpy as np
import pandas as pd
from smartcret.data.generate_synthetic import generate_synthetic_data
from smartcret.data.split import split_dataset
from smartcret.models.trainer import ModelTrainer
from smartcret.models.predictor import ConcretePredictor


@pytest.fixture(scope="module")
def trained_predictor(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("models_test")
    df = generate_synthetic_data(n_samples=120, random_seed=42)
    splits = split_dataset(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)

    trainer = ModelTrainer(config_path="configs/model.yaml")
    result = trainer.train_and_evaluate(
        train_df=splits.train,
        val_df=splits.val,
        test_df=splits.test_reserved,
        version="test_v1",
        output_dir=tmp_dir,
    )

    predictor = ConcretePredictor.load(result.artifact_path)
    return predictor


def test_model_predictions_and_uncertainty(trained_predictor):
    sample_mix = {
        "cemento_kg": 350.0,
        "ceniza_volante_kg": 50.0,
        "escoria_kg": 0.0,
        "puzolana_kg": 0.0,
        "agua_kg": 180.0,
        "superplastificante_kg": 3.0,
        "agregado_grueso_kg": 1050.0,
        "agregado_fino_kg": 750.0,
        "edad_dias": 28,
        "tipo_cemento": "Tipo I",
    }

    pred = trained_predictor.predict(sample_mix)
    assert pred.resistencia_mpa > 15.0
    assert pred.resistencia_lower <= pred.resistencia_mpa <= pred.resistencia_upper
    assert pred.slump_mm >= 0.0
    assert pred.slump_lower <= pred.slump_mm <= pred.slump_upper
    assert not pred.is_extrapolated


def test_extrapolation_detection(trained_predictor):
    # Cement 800 kg/m³ is far beyond training bounds
    extreme_mix = {
        "cemento_kg": 800.0,
        "ceniza_volante_kg": 0.0,
        "escoria_kg": 0.0,
        "puzolana_kg": 0.0,
        "agua_kg": 180.0,
        "superplastificante_kg": 2.0,
        "agregado_grueso_kg": 1000.0,
        "agregado_fino_kg": 700.0,
        "edad_dias": 28,
        "tipo_cemento": "Tipo I",
    }

    pred = trained_predictor.predict(extreme_mix)
    assert pred.is_extrapolated
    assert any("cemento_kg" in msg for msg in pred.extrapolated_features)
