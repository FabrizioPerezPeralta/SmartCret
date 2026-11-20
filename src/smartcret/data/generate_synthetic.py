"""Synthetic concrete mix data generator adhering to physics and domain relationships."""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_samples: int = 500, random_seed: int = 42
) -> pd.DataFrame:
    """Generates synthetic concrete mix proportions and compressive strength / slump

    following established empirical and physical relationships (Abrams' law,
    maturity development, SCM efficiency, and superplasticizer action).
    """
    rng = np.random.default_rng(random_seed)

    # 1. Sample total binder (cementitious material) [kg/m³]
    total_binder = rng.uniform(280.0, 480.0, size=n_samples)

    # 2. Sample cement types and SCM replacement percentages
    cement_types = rng.choice(
        ["Tipo I", "Tipo IP", "Tipo IPM", "Tipo HS", "Tipo V", "Tipo GU"],
        size=n_samples,
        p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05],
    )

    # Proportion of SCMs
    # Scenario: Plain concrete, or binary/ternary blends
    blend_pattern = rng.choice(
        ["plain", "fly_ash", "slag", "pozzolan", "ternary"],
        size=n_samples,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    fly_ash_pct = np.zeros(n_samples)
    slag_pct = np.zeros(n_samples)
    pozzolan_pct = np.zeros(n_samples)

    for i, pattern in enumerate(blend_pattern):
        if pattern == "fly_ash":
            fly_ash_pct[i] = rng.uniform(0.10, 0.35)
        elif pattern == "slag":
            slag_pct[i] = rng.uniform(0.15, 0.50)
        elif pattern == "pozzolan":
            pozzolan_pct[i] = rng.uniform(0.10, 0.30)
        elif pattern == "ternary":
            fly_ash_pct[i] = rng.uniform(0.08, 0.20)
            slag_pct[i] = rng.uniform(0.10, 0.30)

    ceniza_volante = total_binder * fly_ash_pct
    escoria = total_binder * slag_pct
    puzolana = total_binder * pozzolan_pct
    cemento = total_binder - (ceniza_volante + escoria + puzolana)

    # 3. Water-to-cementitious ratio (w/cm) and Water content
    relacion_ac = rng.uniform(0.35, 0.62, size=n_samples)
    agua = total_binder * relacion_ac
    # Ensure water is within realistic normal concrete range [135, 240]
    agua = np.clip(agua, 135.0, 240.0)
    # Recalculate relacion_ac based on final water and binder
    relacion_ac = (agua / total_binder).round(3)

    # 4. Superplasticizer (SP) dosage [kg/m³]
    # Often higher for low w/c mixes
    sp_prob = rng.random(n_samples)
    superplastificante = np.zeros(n_samples)
    for i in range(n_samples):
        if relacion_ac[i] < 0.45 or sp_prob[i] > 0.4:
            # Between 0.5% and 2.5% of binder
            superplastificante[i] = total_binder[i] * rng.uniform(0.005, 0.022)

    # 5. Volumetric calculation for aggregates
    # Specific gravities (typical):
    # Cement: 3.15, Fly Ash: 2.30, Slag: 2.90, Pozzolan: 2.45, Water: 1.00, SP: 1.10
    # Aggregates: 2.65, Air: ~2.0% (20 L)
    v_cemento = cemento / 3150.0
    v_ceniza = ceniza_volante / 2300.0
    v_escoria = escoria / 2900.0
    v_puzolana = puzolana / 2450.0
    v_agua = agua / 1000.0
    v_sp = superplastificante / 1100.0
    v_air = rng.uniform(0.015, 0.025, size=n_samples)

    v_paste_and_air = (
        v_cemento + v_ceniza + v_escoria + v_puzolana + v_agua + v_sp + v_air
    )
    v_agg_total = np.maximum(1.0 - v_paste_and_air, 0.50)  # total aggregate volume in m³

    # Sand-to-aggregate ratio (sand ratio)
    sand_ratio = rng.uniform(0.38, 0.48, size=n_samples)
    v_fine = v_agg_total * sand_ratio
    v_coarse = v_agg_total * (1.0 - sand_ratio)

    rho_agg = 2650.0  # kg/m³
    agregado_fino = v_fine * rho_agg
    agregado_grueso = v_coarse * rho_agg

    # 6. Curing Age [days]
    standard_ages = [3, 7, 14, 28, 56, 90]
    edad_dias = rng.choice(standard_ages, size=n_samples, p=[0.05, 0.25, 0.10, 0.45, 0.10, 0.05])

    # 7. Compressive Strength Calculation (Abrams Law + Maturity + SCM factors)
    # Effective binder calculation at given age:
    # Fly ash and slag gain relative efficiency over time
    k_flyash = np.where(edad_dias <= 7, 0.35, np.where(edad_dias <= 28, 0.75, 1.05))
    k_slag = np.where(edad_dias <= 7, 0.55, np.where(edad_dias <= 28, 0.95, 1.15))
    k_pozz = np.where(edad_dias <= 7, 0.40, np.where(edad_dias <= 28, 0.75, 1.00))

    effective_binder = (
        cemento
        + k_flyash * ceniza_volante
        + k_slag * escoria
        + k_pozz * puzolana
    )
    effective_wc = agua / np.maximum(effective_binder, 100.0)

    # Abrams base formula: fc_28 = A / (B^(w/c))
    A_param = rng.normal(105.0, 3.5, size=n_samples)
    B_param = rng.normal(4.8, 0.15, size=n_samples)
    fc_base_28 = A_param / (B_param ** effective_wc)

    # Age maturity curve: t / (a + b * t) normalized to 28 days
    # At t=28, multiplier = 1.0
    age_multiplier = (edad_dias / (4.0 + 0.857 * edad_dias))

    # Strength with random experimental noise (~3% - 5% CV)
    resistencia = fc_base_28 * age_multiplier * rng.normal(1.0, 0.045, size=n_samples)
    resistencia = np.clip(resistencia, 10.0, 85.0).round(1)

    # 8. Slump (Asentamiento) Calculation [mm]
    # Driven by free water, SP, ball-bearing effect of fly ash, and aggregate proportions
    base_slump = 2.2 * (agua - 140.0)
    sp_boost = 18.0 * superplastificante
    fly_ash_boost = 0.25 * ceniza_volante
    fine_reduction = 0.05 * (agregado_fino - 700.0)
    
    slump = base_slump + sp_boost + fly_ash_boost - fine_reduction + rng.normal(0, 15.0, size=n_samples)
    slump = np.clip(slump, 10.0, 240.0).round(0)

    df = pd.DataFrame(
        {
            "cemento_kg": cemento.round(1),
            "ceniza_volante_kg": ceniza_volante.round(1),
            "escoria_kg": escoria.round(1),
            "puzolana_kg": puzolana.round(1),
            "agua_kg": agua.round(1),
            "superplastificante_kg": superplastificante.round(2),
            "agregado_grueso_kg": agregado_grueso.round(1),
            "agregado_fino_kg": agregado_fino.round(1),
            "edad_dias": edad_dias.astype(int),
            "tipo_cemento": cement_types,
            "relacion_ac": relacion_ac.round(3),
            "resistencia_mpa": resistencia,
            "slump_mm": slump,
            "fuente": "sintetico",
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generador de mezclas de concreto sintéticas para SmartCret."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Número de filas sintéticas a generar (defecto: 500)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/synthetic_concrete.csv",
        help="Ruta del archivo de salida CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria reproducible",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generando {args.n} mezclas sintéticas (semilla={args.seed})...")
    df = generate_synthetic_data(n_samples=args.n, random_seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"Archivo guardado exitosamente en: {out_path} ({len(df)} filas)")


if __name__ == "__main__":
    main()
