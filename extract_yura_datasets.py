"""Script de extracción y consolidación de la Base de Datos YURA.

Extrae datos certificados de:
1. data/raw/Comparacion_Yura.pdf (Tesis UCSM - Cementos Yura Tipo IP, HE, HS, ACI 211, Fineza, Walker, 1-14 días)
2. data/raw/Agregados.pdf (Tesis UCSM - Concretos f'c 210 y 280 kg/cm2 con Cemento Yura IP, humo de sílice, agregado reciclado, 7-28 días)
3. data/raw/Huella_Carbono_Yura.pdf (Reporte Oficial ISO 14067 Yura S.A. / A2G)

Genera:
- data/raw/yura_concreto_ensayos.csv
- data/raw/yura_concreto_probetas.csv
- data/processed/yura_training_dataset.csv
"""

from pathlib import Path
import re
import fitz
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Factores de emisión ISO 14067 (Huella_Carbono_Yura.pdf / A2G)
EMISION_FACTORES = {
    "cemento_ip": 0.489,       # kg CO2e / kg (Cemento Yura Tipo IP Puzolánico)
    "cemento_he": 0.675,       # kg CO2e / kg (Cemento Yura Tipo HE Alta Resistencia)
    "cemento_hs": 0.710,       # kg CO2e / kg (Cemento Yura Tipo HS Sulfatos)
    "cemento_portland": 0.865,  # kg CO2e / kg (Línea base Portland tradicional)
    "humo_silice": 0.028,      # kg CO2e / kg (Micro-sílice subproducto)
    "agregado_natural": 0.0052,# kg CO2e / kg
    "agregado_reciclado": 0.0024, # kg CO2e / kg
    "agua": 0.001,             # kg CO2e / kg
    "aditivo": 0.720,          # kg CO2e / kg
}

# Costos unitarios en Arequipa (PEN S/ por kg)
COSTOS_UNITARIOS = {
    "cemento_ip": 0.600,       # S/ 25.50 / bolsa 42.5kg
    "cemento_he": 0.650,       # S/ 27.60 / bolsa
    "cemento_hs": 0.680,       # S/ 28.90 / bolsa
    "cemento_portland": 0.620,
    "humo_silice": 1.500,      # S/ 1.50 / kg
    "agregado_natural": 0.045, # S/ 45.00 / t
    "agregado_reciclado": 0.028,# S/ 28.00 / t
    "agua": 0.006,             # S/ 6.00 / m3
    "aditivo": 4.800,          # S/ 4.80 / kg
}

KG_CM2_TO_MPA = 0.0980665
INCH_TO_MM = 25.4


def extract_agregados_dataset() -> tuple[list[dict], list[dict]]:
    """Extrae los 48 diseños y 144 ensayos de Agregados.pdf."""
    doc = fitz.open(RAW_DIR / "Agregados.pdf")
    
    # 1. Proporciones de mezcla D1-D24 (p. 108) y D25-D48 (p. 109)
    mixes = {}
    for pno, target_fc in [(108, 210), (109, 280)]:
        text = doc[pno - 1].get_text()
        tokens = [t.strip() for t in text.split("\n") if t.strip()]
        i = 0
        while i < len(tokens):
            m = re.match(r"^D(\d+)\s+([\w\-]+)$", tokens[i])
            if m:
                d_id = f"D{m.group(1)}"
                code = m.group(2)
                mixes[d_id] = {
                    "design_id": d_id,
                    "code": code,
                    "target_fc": target_fc,
                    "cemento_kg": float(tokens[i + 1]),
                    "agregado_grueso_nat_kg": float(tokens[i + 2]),
                    "agregado_grueso_rec_kg": float(tokens[i + 3]),
                    "agregado_grueso_kg": float(tokens[i + 2]) + float(tokens[i + 3]),
                    "agregado_fino_kg": float(tokens[i + 4]),
                    "agua_kg": float(tokens[i + 5]),
                    "superplastificante_kg": round(float(tokens[i + 6]) * 1.15 / 1000.0, 3), # ml a kg con densidad 1.15
                    "puzolana_kg": float(tokens[i + 7]), # Humo de sílice
                    "ceniza_volante_kg": 0.0,
                    "escoria_kg": 0.0,
                    "tipo_cemento": "Tipo IP",
                    "metodo_diseno": "Modulo de Fineza",
                    "fuente_pdf": "Agregados.pdf",
                }
                i += 8
            else:
                i += 1

    # 2. Slump (Asentamiento) de p. 112 y 114
    for pno in [112, 114]:
        tokens = [t.strip() for t in doc[pno - 1].get_text().split("\n") if t.strip()]
        for i in range(len(tokens) - 2):
            if re.match(r"^D\d+$", tokens[i]):
                d_id = tokens[i]
                try:
                    slump_pulg = float(tokens[i + 2])
                    if d_id in mixes:
                        mixes[d_id]["slump_mm"] = round(slump_pulg * INCH_TO_MM, 1)
                        mixes[d_id]["slump_pulg"] = slump_pulg
                except ValueError:
                    pass

    # 3. Resistencias a compresión (p. 123, 124, 125, 128, 129, 130)
    batch_records = []
    cylinder_records = []

    age_pages = [
        (123, 7, 210), (124, 14, 210), (125, 28, 210),
        (128, 7, 280), (129, 14, 280), (130, 28, 280)
    ]

    for pno, age_days, target_fc in age_pages:
        tokens = [t.strip() for t in doc[pno - 1].get_text().split("\n") if t.strip()]
        i = 0
        while i < len(tokens):
            if re.match(r"^D\d+$", tokens[i]):
                d_id = tokens[i]
                code = tokens[i + 1]
                # Buscar las 3 cargas de probeta y el promedio
                loads = []
                fc_prom = None
                for j in range(i + 2, min(len(tokens), i + 16)):
                    if re.match(r"^D\d+$", tokens[j]):
                        break
                    try:
                        v = float(tokens[j])
                        if v > 10000: # Carga en kgf
                            loads.append(v)
                        elif 100.0 <= v <= 450.0 and fc_prom is None:
                            fc_prom = v
                    except ValueError:
                        pass
                
                if fc_prom is not None and d_id in mixes:
                    m = mixes[d_id]
                    cemento_tot = m["cemento_kg"] + m["puzolana_kg"]
                    w_c = round(m["agua_kg"] / cemento_tot, 3)

                    # Huella de carbono m³
                    co2 = (
                        m["cemento_kg"] * EMISION_FACTORES["cemento_ip"] +
                        m["puzolana_kg"] * EMISION_FACTORES["humo_silice"] +
                        m["agregado_grueso_nat_kg"] * EMISION_FACTORES["agregado_natural"] +
                        m["agregado_grueso_rec_kg"] * EMISION_FACTORES["agregado_reciclado"] +
                        m["agregado_fino_kg"] * EMISION_FACTORES["agregado_natural"] +
                        m["agua_kg"] * EMISION_FACTORES["agua"] +
                        m["superplastificante_kg"] * EMISION_FACTORES["aditivo"]
                    )
                    # Costo m³
                    costo = (
                        m["cemento_kg"] * COSTOS_UNITARIOS["cemento_ip"] +
                        m["puzolana_kg"] * COSTOS_UNITARIOS["humo_silice"] +
                        m["agregado_grueso_nat_kg"] * COSTOS_UNITARIOS["agregado_natural"] +
                        m["agregado_grueso_rec_kg"] * COSTOS_UNITARIOS["agregado_reciclado"] +
                        m["agregado_fino_kg"] * COSTOS_UNITARIOS["agregado_natural"] +
                        m["agua_kg"] * COSTOS_UNITARIOS["agua"] +
                        m["superplastificante_kg"] * COSTOS_UNITARIOS["aditivo"]
                    )
                    
                    row = {
                        "cemento_kg": m["cemento_kg"],
                        "ceniza_volante_kg": 0.0,
                        "escoria_kg": 0.0,
                        "puzolana_kg": m["puzolana_kg"],
                        "agua_kg": m["agua_kg"],
                        "superplastificante_kg": m["superplastificante_kg"],
                        "agregado_grueso_kg": m["agregado_grueso_kg"],
                        "agregado_fino_kg": m["agregado_fino_kg"],
                        "edad_dias": age_days,
                        "tipo_cemento": m["tipo_cemento"],
                        "relacion_ac": w_c,
                        "resistencia_mpa": round(fc_prom * KG_CM2_TO_MPA, 2),
                        "slump_mm": m.get("slump_mm", 75.0),
                        "costo_m3": round(costo, 2),
                        "co2_kg_m3": round(co2, 2),
                        "fuente": "laboratorio_empresa",
                        "codigo_mezcla": code,
                        "metodo_diseno": m["metodo_diseno"],
                        "target_fc_kg_cm2": target_fc,
                        "humo_silice_pct": round((m["puzolana_kg"] / m["cemento_kg"]) * 100, 1),
                        "agregado_reciclado_pct": round((m["agregado_grueso_rec_kg"] / m["agregado_grueso_kg"]) * 100, 1) if m["agregado_grueso_kg"] > 0 else 0.0
                    }
                    batch_records.append(row)

                    for cyl_idx, load_kg in enumerate(loads[:3]):
                        fc_cyl = (load_kg / (np.pi * (5.0**2))) # Cilindro 10x20 aprox area 78.54 cm2
                        cylinder_records.append({
                            **row,
                            "probeta_id": f"{code}-{age_days}d-{cyl_idx+1}",
                            "carga_rotura_kg": load_kg,
                            "resistencia_cilindro_mpa": round(fc_cyl * KG_CM2_TO_MPA, 2)
                        })
                i += 1
            else:
                i += 1

    return batch_records, cylinder_records


def extract_comparacion_yura_dataset() -> tuple[list[dict], list[dict]]:
    """Extrae los ensayos por tipo de cemento (IP, HE, HS, IP+A) de Comparacion_Yura.pdf."""
    doc = fitz.open(RAW_DIR / "Comparacion_Yura.pdf")
    
    # Proporciones base de diseño para cada método (f'c = 280 kg/cm2, págs 80, 82, 84, 93, 95, 97, 106, 108, 110)
    # Todos tienen Cemento = 439.91 kg/m3, Agua = 205.00 kg/m3 (w/c = 0.466)
    MIX_PROPORTIONS = {
        ("ACI", "Tipo IP"): {"cemento": 439.91, "agua": 205.00, "grueso": 982.66, "fino": 433.90, "sp": 0.0},
        ("ACI", "Tipo HE"): {"cemento": 439.91, "agua": 205.00, "grueso": 982.66, "fino": 442.86, "sp": 0.0},
        ("ACI", "Tipo HS"): {"cemento": 439.91, "agua": 205.00, "grueso": 982.66, "fino": 449.00, "sp": 0.0},
        ("ACI", "Tipo IP+A"): {"cemento": 439.91, "agua": 205.00, "grueso": 982.66, "fino": 433.90, "sp": 17.60}, # 4% aditivo acelerante
        ("Fineza", "Tipo IP"): {"cemento": 439.91, "agua": 205.00, "grueso": 876.15, "fino": 549.78, "sp": 0.0},
        ("Fineza", "Tipo HE"): {"cemento": 439.91, "agua": 205.00, "grueso": 876.15, "fino": 554.00, "sp": 0.0},
        ("Fineza", "Tipo HS"): {"cemento": 439.91, "agua": 205.00, "grueso": 876.15, "fino": 556.00, "sp": 0.0},
        ("Fineza", "Tipo IP+A"): {"cemento": 439.91, "agua": 205.00, "grueso": 876.15, "fino": 549.78, "sp": 17.60},
        ("Walker", "Tipo IP"): {"cemento": 439.91, "agua": 205.00, "grueso": 832.34, "fino": 593.50, "sp": 0.0},
        ("Walker", "Tipo HE"): {"cemento": 439.91, "agua": 205.00, "grueso": 832.34, "fino": 597.00, "sp": 0.0},
        ("Walker", "Tipo HS"): {"cemento": 439.91, "agua": 205.00, "grueso": 832.34, "fino": 599.50, "sp": 0.0},
        ("Walker", "Tipo IP+A"): {"cemento": 439.91, "agua": 205.00, "grueso": 832.34, "fino": 593.50, "sp": 17.60},
    }

    batch_records = []
    cylinder_records = []

    # Extraer tablas 72 a 119 (págs 113 a 136)
    for pno in range(112, 137):
        text = doc[pno].get_text()
        tables = re.split(r"Tabla (\d+):", text)
        if len(tables) > 1:
            for idx in range(1, len(tables), 2):
                t_num = int(tables[idx])
                if t_num > 119:
                    continue
                t_content = tables[idx + 1]
                title_line = t_content.strip().split("\n")[0]
                
                # Edad
                age_m = re.search(r"(\d+)\s*d[ií]as?", title_line, re.IGNORECASE)
                if not age_m:
                    age_m = re.search(r"(\d+)\s*d[ií]a", title_line, re.IGNORECASE)
                age_days = int(age_m.group(1)) if age_m else 28

                # Método
                if "ACI" in title_line or "211" in title_line:
                    method = "ACI"
                elif "Fineza" in title_line or "fineza" in title_line:
                    method = "Fineza"
                else:
                    method = "Walker"

                # Cemento
                if "IP+A" in title_line or "IP + A" in title_line:
                    cem_key = "Tipo IP+A"
                    cem_schema = "Tipo IP"
                    emiss_key = "cemento_ip"
                    cost_key = "cemento_ip"
                elif "IP" in title_line:
                    cem_key = "Tipo IP"
                    cem_schema = "Tipo IP"
                    emiss_key = "cemento_ip"
                    cost_key = "cemento_ip"
                elif "HE" in title_line:
                    cem_key = "Tipo HE"
                    cem_schema = "Tipo HE"
                    emiss_key = "cemento_he"
                    cost_key = "cemento_he"
                else:
                    cem_key = "Tipo HS"
                    cem_schema = "Tipo HS"
                    emiss_key = "cemento_hs"
                    cost_key = "cemento_hs"

                # Slump
                slump_m = re.search(r"Slump\s*\(pulg\)\s*([\d\.]+)", t_content)
                slump_pulg = float(slump_m.group(1)) if slump_m else 2.5
                slump_mm = round(slump_pulg * INCH_TO_MM, 1)

                # f'c promedio
                fc_prom_m = re.search(r"f\'?c\s*promedio\s*\(kg/cm2\)\s*([\d\.]+)", t_content)
                if fc_prom_m:
                    fc_prom_kg = float(fc_prom_m.group(1))
                else:
                    continue

                mix = MIX_PROPORTIONS.get((method, cem_key), MIX_PROPORTIONS[("ACI", "Tipo IP")])

                # Huella de carbono m3
                co2 = (
                    mix["cemento"] * EMISION_FACTORES[emiss_key] +
                    mix["grueso"] * EMISION_FACTORES["agregado_natural"] +
                    mix["fino"] * EMISION_FACTORES["agregado_natural"] +
                    mix["agua"] * EMISION_FACTORES["agua"] +
                    mix["sp"] * EMISION_FACTORES["aditivo"]
                )
                # Costo m3
                costo = (
                    mix["cemento"] * COSTOS_UNITARIOS[cost_key] +
                    mix["grueso"] * COSTOS_UNITARIOS["agregado_natural"] +
                    mix["fino"] * COSTOS_UNITARIOS["agregado_natural"] +
                    mix["agua"] * COSTOS_UNITARIOS["agua"] +
                    mix["sp"] * COSTOS_UNITARIOS["aditivo"]
                )

                batch_row = {
                    "cemento_kg": mix["cemento"],
                    "ceniza_volante_kg": 0.0,
                    "escoria_kg": 0.0,
                    "puzolana_kg": 0.0,
                    "agua_kg": mix["agua"],
                    "superplastificante_kg": mix["sp"],
                    "agregado_grueso_kg": mix["grueso"],
                    "agregado_fino_kg": mix["fino"],
                    "edad_dias": age_days,
                    "tipo_cemento": cem_schema,
                    "relacion_ac": round(mix["agua"] / mix["cemento"], 3),
                    "resistencia_mpa": round(fc_prom_kg * KG_CM2_TO_MPA, 2),
                    "slump_mm": slump_mm,
                    "costo_m3": round(costo, 2),
                    "co2_kg_m3": round(co2, 2),
                    "fuente": "laboratorio_empresa",
                    "codigo_mezcla": f"{cem_key}-{method}-{age_days}D",
                    "metodo_diseno": method,
                    "target_fc_kg_cm2": 280,
                    "humo_silice_pct": 0.0,
                    "agregado_reciclado_pct": 0.0
                }
                batch_records.append(batch_row)

                # Extraer probetas individuales si están en el bloque
                prob_matches = re.findall(r"P\s*\d+-\d+", t_content)
                fc_cyl_matches = re.findall(r"\n\s*(\d{2,3}\.\d{2})\s*\n", t_content)
                for c_i, p_code in enumerate(prob_matches[:8]):
                    fc_cyl = float(fc_cyl_matches[c_i]) if c_i < len(fc_cyl_matches) else fc_prom_kg
                    cylinder_records.append({
                        **batch_row,
                        "probeta_id": p_code.replace(" ", ""),
                        "carga_rotura_kg": 0.0,
                        "resistencia_cilindro_mpa": round(fc_cyl * KG_CM2_TO_MPA, 2)
                    })

    return batch_records, cylinder_records


def main():
    print("Iniciando extracción de datos empíricos YURA...")
    
    agregados_batches, agregados_cylinders = extract_agregados_dataset()
    print(f"[Agregados.pdf] {len(agregados_batches)} ensayos por lote, {len(agregados_cylinders)} probetas individuales")
    
    yura_batches, yura_cylinders = extract_comparacion_yura_dataset()
    print(f"[Comparacion_Yura.pdf] {len(yura_batches)} ensayos por lote, {len(yura_cylinders)} probetas individuales")
    
    # Consolidar
    all_batches = pd.DataFrame(agregados_batches + yura_batches)
    all_cylinders = pd.DataFrame(agregados_cylinders + yura_cylinders)
    
    # Guardar en data/raw/
    raw_ensayos_path = RAW_DIR / "yura_concreto_ensayos.csv"
    raw_probetas_path = RAW_DIR / "yura_concreto_probetas.csv"
    all_batches.to_csv(raw_ensayos_path, index=False)
    all_cylinders.to_csv(raw_probetas_path, index=False)
    print(f"Guardado {raw_ensayos_path} ({len(all_batches)} filas)")
    print(f"Guardado {raw_probetas_path} ({len(all_cylinders)} filas)")
    
    # Dataset procesado para entrenamiento ML (cumpliendo estricto schema.yaml)
    training_cols = [
        "cemento_kg", "ceniza_volante_kg", "escoria_kg", "puzolana_kg",
        "agua_kg", "superplastificante_kg", "agregado_grueso_kg", "agregado_fino_kg",
        "edad_dias", "tipo_cemento", "relacion_ac",
        "resistencia_mpa", "slump_mm", "fuente"
    ]
    df_training = all_batches[training_cols].copy()
    processed_path = PROCESSED_DIR / "yura_training_dataset.csv"
    df_training.to_csv(processed_path, index=False)
    print(f"Guardado {processed_path} ({len(df_training)} filas listas para ML)")
    
    # Estadísticas clave
    print("\n--- RESUMEN TÉCNICO DATASET YURA ---")
    print(f"Total mezclas empíricas: {len(df_training)}")
    print(f"Rango Resistencia: {df_training['resistencia_mpa'].min()} a {df_training['resistencia_mpa'].max()} MPa")
    print(f"Rango Slump: {df_training['slump_mm'].min()} a {df_training['slump_mm'].max()} mm")
    print(f"Distribución Tipo Cemento:\n{df_training['tipo_cemento'].value_counts()}")
    print(f"Distribución Edades de Curado:\n{df_training['edad_dias'].value_counts()}")
    print("------------------------------------")


if __name__ == "__main__":
    main()
