"""Scenario persistence (SQLite), comparative evaluation, and PDF executive reports."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from smartcret.optimizer.optimizer import MixProposal


@dataclass
class ConcreteScenario:
    scenario_id: str
    name: str
    project_name: str
    requirements: Dict[str, Any]
    mix: MixProposal
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ScenarioManager:
    """Stores, reloads, and compares concrete design scenarios with PDF export."""

    def __init__(self, db_path: str = "smartcret.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenarios (
                    scenario_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    requirements_json TEXT NOT NULL,
                    mix_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_scenario(self, scenario: ConcreteScenario) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scenarios 
                (scenario_id, name, project_name, requirements_json, mix_json, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario.scenario_id,
                    scenario.name,
                    scenario.project_name,
                    json.dumps(scenario.requirements),
                    json.dumps(asdict(scenario.mix)),
                    scenario.notes,
                    scenario.created_at,
                ),
            )
            conn.commit()

    def get_scenario(self, scenario_id: str) -> ConcreteScenario:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM scenarios WHERE scenario_id = ?", (scenario_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Escenario '{scenario_id}' no encontrado.")

            reqs = json.loads(row[3])
            mix_dict = json.loads(row[4])
            mix = MixProposal(**mix_dict)

            return ConcreteScenario(
                scenario_id=row[0],
                name=row[1],
                project_name=row[2],
                requirements=reqs,
                mix=mix,
                notes=row[5],
                created_at=row[6],
            )

    def list_scenarios(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT scenario_id, name, project_name, mix_json, created_at FROM scenarios ORDER BY created_at DESC"
            )
            rows = cur.fetchall()

        results = []
        for r in rows:
            mix = json.loads(r[3])
            results.append(
                {
                    "scenario_id": r[0],
                    "name": r[1],
                    "project_name": r[2],
                    "costo_s_m3": round(mix.get("costo_soles_m3", 0.0), 2),
                    "co2_kg_m3": round(mix.get("co2_kg_m3", 0.0), 1),
                    "fc_pred_mpa": round(mix.get("fc_pred_mpa", 0.0), 1),
                    "created_at": r[4],
                }
            )
        return results

    def compare_scenarios(self, scenario_ids: List[str]) -> pd.DataFrame:
        """Compares multiple scenarios side by side, computing differences."""
        scenarios = [self.get_scenario(sid) for sid in scenario_ids]
        if not scenarios:
            return pd.DataFrame()

        rows = []
        for s in scenarios:
            m = s.mix
            rows.append(
                {
                    "Escenario": s.name,
                    "Proyecto": s.project_name,
                    "Perfil": m.perfil,
                    "Costo (S/ / m³)": round(m.costo_soles_m3, 2),
                    "Emisión (kg CO₂ / m³)": round(m.co2_kg_m3, 1),
                    "f'c Estimada (MPa)": f"{m.fc_pred_mpa:.1f} [{m.fc_lower_mpa:.1f} - {m.fc_upper_mpa:.1f}]",
                    "Slump (mm)": f"{m.slump_pred_mm:.0f} [{m.slump_lower_mm:.0f} - {m.slump_upper_mm:.0f}]",
                    "Cemento (kg)": round(m.cemento_kg, 1),
                    "Ceniza Volante (kg)": round(m.ceniza_volante_kg, 1),
                    "Escoria (kg)": round(m.escoria_kg, 1),
                    "Puzolana (kg)": round(m.puzolana_kg, 1),
                    "Agua (kg)": round(m.agua_kg, 1),
                    "Superplast. (kg)": round(m.superplastificante_kg, 2),
                    "Agregado Grueso (kg)": round(m.agregado_grueso_kg, 1),
                    "Agregado Fino (kg)": round(m.agregado_fino_kg, 1),
                    "a/cm": round(m.relacion_ac, 3),
                    "Extrapolación": "Sí ⚠️" if m.es_extrapolacion else "No (Segura)",
                }
            )

        df = pd.DataFrame(rows).set_index("Escenario").T
        return df

    def export_pdf_report(
        self, scenario_ids: List[str], output_path: Union[str, Path]
    ) -> str:
        """Generates an executive PDF report comparing scenarios using ReportLab."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        scenarios = [self.get_scenario(sid) for sid in scenario_ids]
        doc = SimpleDocTemplate(str(out), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=8,
        )
        subtitle_style = ParagraphStyle(
            "SubTitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=14,
        )

        story.append(Paragraph("SMARTCRET: FICHA TÉCNICA COMPARATIVA DE ESCENARIOS", title_style))
        story.append(
            Paragraph(
                f"Diseño y Optimización de Mezclas de Concreto Ecoeficientes | Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                subtitle_style,
            )
        )
        story.append(Spacer(1, 10))

        # Build comparison data
        comp_df = self.compare_scenarios(scenario_ids)
        table_data = [["Variable / Componente"] + list(comp_df.columns)]
        for idx, row in comp_df.iterrows():
            table_data.append([str(idx)] + [str(v) for v in row.values])

        t = Table(table_data, colWidths=[150] + [380 // max(len(comp_df.columns), 1)] * len(comp_df.columns))
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 16))

        # Notes section
        notes_text = (
            "<b>Notas Técnicas y Declaración de Incertidumbre:</b><br/>"
            "1. Las mezclas reportadas han sido optimizadas mediante algoritmo genético multiobjetivo (NSGA-II) "
            "sujeto a las restricciones y requisitos de durabilidad del código estructural ACI 318 y ACI 211.1.<br/>"
            "2. Los intervalos de confianza reportados corresponden al 90% (calibración conforme). Toda mezcla "
            "propuesta debe validarse empíricamente en laboratorio antes de su aplicación en elementos estructurales.<br/>"
            "3. Los factores de emisión de CO₂ corresponden a declaraciones ambientales de producto (EPD) promedio."
        )
        story.append(Paragraph(notes_text, styles["Normal"]))

        doc.build(story)
        return str(out)
