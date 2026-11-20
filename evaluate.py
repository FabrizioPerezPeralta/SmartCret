"""Evaluation script for trained SmartCret models on reserved test data."""

import argparse
from pathlib import Path
import sys

# Ensure 'src' is in sys.path
_src_dir = Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import numpy as np
import pandas as pd

from smartcret.models.predictor import ConcretePredictor
from smartcret.data.loader import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluación del modelo SmartCret sobre datos de test reservado."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/v1/predictor.pkl",
        help="Ruta al modelo entrenado (.pkl)",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/test_reserved.csv",
        help="Ruta al archivo CSV de test reservado",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    test_path = Path(args.test_data)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. Ejecuta 'python train.py' primero."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Archivo de test no encontrado en {test_path}."
        )

    print("=" * 60)
    print("🧪 EVALUACIÓN DE MODELO SMARTCRET SOBRE DATOS RESERVADOS")
    print(f"Modelo: {model_path}")
    print(f"Datos:  {test_path}")
    print("=" * 60)

    predictor = ConcretePredictor.load(model_path)
    df_test = pd.read_csv(test_path)

    predictions = predictor.predict(df_test)

    # Compute metrics
    y_true_res = df_test["resistencia_mpa"].values
    y_pred_res = np.array([p.resistencia_mpa for p in predictions])
    res_lower = np.array([p.resistencia_lower for p in predictions])
    res_upper = np.array([p.resistencia_upper for p in predictions])

    y_true_slump = df_test["slump_mm"].values
    y_pred_slump = np.array([p.slump_mm for p in predictions])
    slump_lower = np.array([p.slump_lower for p in predictions])
    slump_upper = np.array([p.slump_upper for p in predictions])

    # Coverage of 90% prediction intervals
    res_coverage = np.mean((y_true_res >= res_lower) & (y_true_res <= res_upper)) * 100
    slump_coverage = np.mean((y_true_slump >= slump_lower) & (y_true_slump <= slump_upper)) * 100

    res_mape = np.mean(np.abs((y_true_res - y_pred_res) / y_true_res)) * 100
    slump_mape = np.mean(np.abs((y_true_slump - y_pred_slump) / np.maximum(y_true_slump, 1.0))) * 100

    res_rmse = np.sqrt(np.mean((y_true_res - y_pred_res) ** 2))
    slump_rmse = np.sqrt(np.mean((y_true_slump - y_pred_slump) ** 2))

    print("\n📈 DESEMPEÑO EN RESISTENCIA (MPa):")
    print(f"  • RMSE:                {res_rmse:.2f} MPa")
    print(f"  • MAPE:                {res_mape:.2f}% (Meta < 10%: {'CUMPLIDA ✅' if res_mape < 10 else 'NO CUMPLIDA ⚠️'})")
    print(f"  • Cobertura Intervalo: {res_coverage:.1f}% (Objetivo teórico: 90%)")

    print("\n📈 DESEMPEÑO EN SLUMP / TRABAJABILIDAD (mm):")
    print(f"  • RMSE:                {slump_rmse:.1f} mm")
    print(f"  • MAPE:                {slump_mape:.2f}%")
    print(f"  • Cobertura Intervalo: {slump_coverage:.1f}% (Objetivo teórico: 90%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
