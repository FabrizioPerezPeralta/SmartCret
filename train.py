"""Main training entry point for SmartCret models."""

import argparse
from pathlib import Path
import sys

# Ensure 'src' is in sys.path
_src_dir = Path(__file__).resolve().parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import pandas as pd

from smartcret.data.loader import DataLoader
from smartcret.data.generate_synthetic import generate_synthetic_data
from smartcret.data.split import split_dataset, save_splits
from smartcret.models.trainer import ModelTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento y versionado de modelos SmartCret."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/",
        help="Ruta a datos de entrada (directorio o archivo CSV/Excel)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Ruta al archivo YAML de configuración del modelo",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Identificador de versión del modelo (ej. v1, v2)",
    )
    return parser.parse_args()


def load_or_fallback_data(data_path_str: str) -> pd.DataFrame:
    loader = DataLoader("configs/schema.yaml")
    data_path = Path(data_path_str)

    # 1. Check if raw directory has data files
    found_files = []
    if data_path.exists():
        if data_path.is_file():
            found_files = [data_path]
        else:
            found_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.xlsx"))

    if found_files:
        print(f"[SmartCret] Cargando {len(found_files)} archivos de datos desde {data_path}...")
        df, report = loader.load_directory(data_path) if data_path.is_dir() else loader.load_file(data_path)
        print(f"[SmartCret] Datos cargados: {len(df)} filas válidas. Duplicados eliminados: {report.duplicates_removed}")
        return df

    # 2. Check if synthetic dataset exists
    synthetic_path = Path("data/synthetic/synthetic_concrete.csv")
    if not synthetic_path.exists():
        print("[SmartCret] No se encontraron datos en data/raw/. Generando datos sintéticos iniciales...")
        df_syn = generate_synthetic_data(n_samples=500, random_seed=42)
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        df_syn.to_csv(synthetic_path, index=False)
        print(f"[SmartCret] Creado {synthetic_path} con 500 filas sintéticas.")
    else:
        print(f"[SmartCret] No se encontraron datos reales en {data_path}. Utilizando datos sintéticos: {synthetic_path}")

    df, report = loader.load_file(synthetic_path)
    print(f"[SmartCret] Datos sintéticos validados: {len(df)} filas.")
    return df


def main():
    args = parse_args()
    print("=" * 65)
    print(f"🚀 INICIANDO ENTRENAMIENTO SMARTCRET ({args.version})")
    print("=" * 65)

    # 1. Cargar y validar datos
    df = load_or_fallback_data(args.data)

    # 2. Partición Train / Val / Test reservado
    print("\n[1/3] Realizando partición de datos (70% Train, 15% Val, 15% Test reservado)...")
    splits = split_dataset(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42)
    save_splits(splits, output_dir="data/processed")
    print(splits.summary())

    # 3. Entrenamiento y comparación de modelos
    print(f"\n[2/3] Entrenando y comparando modelos con validación cruzada de 5 pliegues...")
    trainer = ModelTrainer(config_path=args.config)
    result = trainer.train_and_evaluate(
        train_df=splits.train,
        val_df=splits.val,
        test_df=splits.test_reserved,
        version=args.version,
        output_dir="models",
    )

    # 4. Reporte de resultados
    print("\n" + "=" * 65)
    print("🏆 RESULTADOS DE LA COMPARACIÓN DE MODELOS (CV):")
    print("=" * 65)
    print(result.comparison_table.to_string(index=False))

    print(f"\n🌟 MODELO GANADOR: {result.best_model_name}")
    print("\n📊 MÉTRICAS EN DATOS RESERVADOS (TEST):")
    print(f"  • Resistencia (MPa):")
    print(f"      - R²:   {result.metrics_resistencia.r2:.4f}")
    print(f"      - RMSE: {result.metrics_resistencia.rmse:.2f} MPa")
    print(f"      - MAE:  {result.metrics_resistencia.mae:.2f} MPa")
    print(f"      - MAPE: {result.metrics_resistencia.mape * 100:.2f}% (Meta: < 10%)")
    print(f"  • Slump (mm):")
    print(f"      - R²:   {result.metrics_slump.r2:.4f}")
    print(f"      - RMSE: {result.metrics_slump.rmse:.1f} mm")
    print(f"      - MAE:  {result.metrics_slump.mae:.1f} mm")
    print(f"      - MAPE: {result.metrics_slump.mape * 100:.2f}%")

    print(f"\n💾 Artefactos guardados exitosamente en:")
    print(f"   -> Modelo serializado: {result.artifact_path}")
    print(f"   -> Metadatos y métricas: models/{args.version}/metadata.json")
    print("=" * 65)
    print("✅ ENTRENAMIENTO COMPLETADO CON ÉXITO")
    print("=" * 65)


if __name__ == "__main__":
    main()
