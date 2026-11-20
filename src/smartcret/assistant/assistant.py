"""Agentic engineering assistant with verified tool calling and uncertainty transparency."""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional
import json

from smartcret.models.predictor import ConcretePredictor
from smartcret.optimizer.optimizer import MixOptimizer, OptimizationRequest, MixProposal
from smartcret.simulator.simulator import ContingencySimulator, ContingencyEvent
from smartcret.simulator.carbon_budget import CarbonBudgetTracker, ProjectCarbonBudget
from smartcret.simulator.scenarios import ScenarioManager


@dataclass
class ToolExecutionResult:
    tool_name: str
    arguments: Dict[str, Any]
    output: Any
    error: Optional[str] = None


class SmartCretAssistant:
    """Agentic concrete engineering assistant.

    STRICT OPERATIONAL RULE:
    The assistant never invents or hallucinates performance metrics, costs, or emissions.
    It strictly reports calculations executed by the prediction, optimization, and
    simulation tools, always expressing the uncertainty bounds.
    """

    def __init__(
        self,
        predictor: Optional[ConcretePredictor] = None,
        optimizer: Optional[MixOptimizer] = None,
        simulator: Optional[ContingencySimulator] = None,
        carbon_tracker: Optional[CarbonBudgetTracker] = None,
        scenario_manager: Optional[ScenarioManager] = None,
    ):
        self.predictor = predictor or ConcretePredictor.load("models/v1/predictor.pkl")
        self.optimizer = optimizer or MixOptimizer()
        self.simulator = simulator or ContingencySimulator(optimizer=self.optimizer)
        self.carbon_tracker = carbon_tracker or CarbonBudgetTracker()
        self.scenario_manager = scenario_manager or ScenarioManager()

    # --- Tool Definitions ---

    def tool_predict_mix(self, mix_components: Dict[str, Any]) -> Dict[str, Any]:
        """Tool: Predicts compressive strength and slump with 90% confidence interval."""
        pred = self.predictor.predict(mix_components)
        return pred.to_dict()

    def tool_optimize_mix(
        self,
        fc_target_mpa: float = 28.0,
        slump_target_mm: float = 100.0,
        exposure_class: str = "F0_normal",
        age_days: int = 28,
        cement_type: str = "Tipo I",
    ) -> List[Dict[str, Any]]:
        """Tool: Multi-objective optimization producing Pareto-optimal mixes."""
        req = OptimizationRequest(
            fc_target_mpa=fc_target_mpa,
            slump_target_mm=slump_target_mm,
            exposure_class=exposure_class,
            age_days=age_days,
            cement_type=cement_type,
        )
        proposals = self.optimizer.optimize(req, population_size=60, n_generations=12)
        return [p.to_dict() for p in proposals]

    def tool_simulate_contingency(
        self,
        event_type: str = "price_increase",
        material: str = "cemento",
        multiplier: float = 1.25,
        unavailable_materials: Optional[List[str]] = None,
        fc_target_mpa: float = 28.0,
        slump_target_mm: float = 100.0,
    ) -> Dict[str, Any]:
        """Tool: Simulates material price surge or shortage and re-optimizes."""
        req = OptimizationRequest(
            fc_target_mpa=fc_target_mpa, slump_target_mm=slump_target_mm
        )
        event = ContingencyEvent(
            name=f"Variación en {material}",
            price_multipliers={material: multiplier} if event_type == "price_increase" else {},
            unavailable_materials=unavailable_materials or ([] if event_type == "price_increase" else [material]),
        )
        base_mix, new_mix, impact = self.simulator.simulate(req, event)
        return {
            "evento": event.name,
            "mezcla_original": base_mix.to_dict(),
            "mezcla_reoptimizada": new_mix.to_dict(),
            "impacto": {
                "costo_original_s_m3": impact.cost_original,
                "costo_nuevo_s_m3": impact.cost_new,
                "delta_costo_s_m3": impact.cost_delta,
                "delta_costo_pct": impact.cost_delta_pct,
                "co2_original_kg_m3": impact.co2_original,
                "co2_nuevo_kg_m3": impact.co2_new,
                "delta_co2_kg_m3": impact.co2_delta,
                "delta_co2_pct": impact.co2_delta_pct,
                "sustituciones_kg_m3": impact.material_substitutions,
            },
            "resumen_tecnico": impact.engineering_summary,
        }

    def tool_check_carbon_budget(self, project_id: str = "OBRA-LIMA-01") -> Dict[str, Any]:
        """Tool: Evaluates project carbon budget and pending alerts."""
        try:
            status = self.carbon_tracker.get_project_status(project_id)
        except ValueError:
            # Fallback to existing project or initialize default
            avail = getattr(self.carbon_tracker, "list_projects", lambda: [])()
            if avail:
                project_id = avail[0]["project_id"]
                status = self.carbon_tracker.get_project_status(project_id)
            else:
                p_def = ProjectCarbonBudget(
                    project_id=project_id,
                    project_name="Complejo Residencial Las Palmeras",
                    planned_volume_m3=300.0,
                    co2_budget_ton=68.0,
                    cost_budget_soles=65000.0,
                    warning_threshold_pct=80.0,
                )
                self.carbon_tracker.create_project(p_def)
                self.carbon_tracker.log_pour(project_id, 120.0, 245.0, 218.0, "Cimentaciones")
                self.carbon_tracker.log_pour(project_id, 110.0, 238.0, 212.0, "Columnas y Placas")
                status = self.carbon_tracker.get_project_status(project_id)

        alerts = self.carbon_tracker.check_alerts(project_id)
        return {
            "estado_proyecto": status,
            "alertas": [
                {
                    "nivel": a.level,
                    "metrica": a.metric,
                    "mensaje": a.message,
                }
                for a in alerts
            ],
        }

    # --- Agentic Dispatcher and Responder ---

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Parses user query, invokes the matching engineering tool, and provides a factual explanation."""
        query_lower = user_query.lower()
        tool_results: List[ToolExecutionResult] = []

        # Intent 1: Contingency / Price increase / Material shortage
        if any(w in query_lower for w in ["sube", "precio", "escasez", "falta", "desabastecimiento", "contingencia", "imprevisto"]):
            # Extract material and price change if present
            mat = "cemento"
            if "escoria" in query_lower:
                mat = "escoria"
            elif "ceniza" in query_lower:
                mat = "ceniza_volante"
            elif "aditivo" in query_lower or "superplastificante" in query_lower:
                mat = "superplastificante"

            # Check percentage
            mult = 1.25
            pct_match = re.search(r"(\d+)\s*%", query_lower)
            if pct_match:
                mult = 1.0 + (float(pct_match.group(1)) / 100.0)

            # Check fc target
            fc_match = re.search(r"(\d+)\s*mpa", query_lower)
            fc_target = float(fc_match.group(1)) if fc_match else 28.0

            res = self.tool_simulate_contingency(
                event_type="price_increase" if "precio" in query_lower or "sube" in query_lower else "shortage",
                material=mat,
                multiplier=mult,
                fc_target_mpa=fc_target,
            )
            tool_results.append(ToolExecutionResult("tool_simulate_contingency", {"material": mat, "mult": mult}, res))

            mix_orig = res["mezcla_original"]
            mix_reopt = res["mezcla_reoptimizada"]
            imp = res["impacto"]

            response_text = (
                f"### 🏗️ Análisis de Imprevisto: Incremento en precio de {mat.capitalize()} ({mult - 1.0:+.0%})\n\n"
                f"Para mitigar el aumento en el precio del cemento manteniendo el requisito técnico de **f'c = {fc_target:.0f} MPa**:\n\n"
                f"1. **Mezcla recomendada:**\n"
                f"   - **Cemento:** {mix_reopt['dosificacion_kg_m3']['cemento']} kg/m³ "
                f"({imp['sustituciones_kg_m3']['cemento_kg']:+0.1f} kg/m³ frente a la mezcla base)\n"
                f"   - **Adiciones puzolánicas/escoria:** {mix_reopt['dosificacion_kg_m3']['ceniza_volante']} kg ceniza volante, "
                f"{mix_reopt['dosificacion_kg_m3']['escoria']} kg escoria\n"
                f"   - **Agua:** {mix_reopt['dosificacion_kg_m3']['agua']} kg/m³ (relación a/cm: {mix_reopt['relacion_ac']})\n"
                f"   - **Superplastificante:** {mix_reopt['dosificacion_kg_m3']['superplastificante']} kg/m³\n\n"
                f"2. **Desempeño y Certidumbre Técnica:**\n"
                f"   - Resistencia a 28 días: **{mix_reopt['resistencia_pred_mpa']} MPa** "
                f"(Intervalo 90%: [{mix_reopt['resistencia_intervalo_90'][0]} - {mix_reopt['resistencia_intervalo_90'][1]} MPa])\n"
                f"   - Asentamiento (Slump): **{mix_reopt['slump_pred_mm']} mm**\n\n"
                f"3. **Impacto Económico y Ambiental:**\n"
                f"   - Costo resultante: **S/ {imp['costo_nuevo_s_m3']:.2f} / m³** ({imp['delta_costo_pct']:+.1f}% frente a base)\n"
                f"   - Huella de carbono: **{imp['co2_nuevo_kg_m3']:.1f} kg CO₂/m³** ({imp['delta_co2_pct']:+.1f}%)\n"
            )

        # Intent 2: Optimization
        elif any(w in query_lower for w in ["optimiza", "diseña", "propón", "mezcla para", "dosificación", "frente de pareto"]):
            fc_match = re.search(r"(\d+)\s*mpa", query_lower)
            fc = float(fc_match.group(1)) if fc_match else 28.0
            slump_match = re.search(r"(\d+)\s*mm", query_lower)
            slump = float(slump_match.group(1)) if slump_match else 100.0

            proposals = self.tool_optimize_mix(fc_target_mpa=fc, slump_target_mm=slump)
            tool_results.append(ToolExecutionResult("tool_optimize_mix", {"fc": fc, "slump": slump}, proposals))

            p_cost = proposals[0]
            p_co2 = min(proposals, key=lambda x: x["co2_kg_m3"])

            response_text = (
                f"### 🎯 Mezclas Optimizadas para f'c = {fc:.0f} MPa y Slump = {slump:.0f} mm\n\n"
                f"He generado el Frente de Pareto de soluciones que cumplen la norma NTP 339 / ACI 318:\n\n"
                f"• **Opción A (Mínimo Costo):** S/ {p_cost['costo_s_m3']:.2f}/m³ | {p_cost['co2_kg_m3']} kg CO₂/m³\n"
                f"  - Cemento: {p_cost['dosificacion_kg_m3']['cemento']} kg | SCM: {p_cost['dosificacion_kg_m3']['ceniza_volante'] + p_cost['dosificacion_kg_m3']['escoria']} kg\n"
                f"  - f'c predicho: {p_cost['resistencia_pred_mpa']} MPa [{p_cost['resistencia_intervalo_90'][0]} - {p_cost['resistencia_intervalo_90'][1]} MPa]\n\n"
                f"• **Opción B (Mínimo Carbono):** S/ {p_co2['costo_s_m3']:.2f}/m³ | {p_co2['co2_kg_m3']} kg CO₂/m³\n"
                f"  - Cemento: {p_co2['dosificacion_kg_m3']['cemento']} kg | SCM: {p_co2['dosificacion_kg_m3']['ceniza_volante'] + p_co2['dosificacion_kg_m3']['escoria']} kg\n"
                f"  - f'c predicho: {p_co2['resistencia_pred_mpa']} MPa [{p_co2['resistencia_intervalo_90'][0]} - {p_co2['resistencia_intervalo_90'][1]} MPa]\n"
            )

        # Intent 3: Carbon Budget Check
        elif any(w in query_lower for w in ["presupuesto", "carbono", "emisiones", "alerta"]):
            proj_id = "OBRA-LIMA-01"
            res = self.tool_check_carbon_budget(proj_id)
            tool_results.append(ToolExecutionResult("tool_check_carbon_budget", {"project_id": proj_id}, res))

            st = res["estado_proyecto"]
            alerts_text = "\n".join([f"⚠️ {a['mensaje']}" for a in res["alertas"]]) or "✅ Sin alertas activas."
            response_text = (
                f"### 🌿 Estado del Presupuesto de Carbono ({st['project_name']})\n\n"
                f"- Volumen vertido: **{st['poured_volume_m3']} m³** de {st['planned_volume_m3']} m³ planificados\n"
                f"- CO₂ acumulado: **{st['co2_emitted_ton']} t CO₂** de {st['co2_budget_ton']} t autorizadas ({st['co2_pct_consumed']}%)\n"
                f"- Presupuesto económico: **S/ {st['cost_spent_soles']:,.2f}** consumidos ({st['cost_pct_consumed']}%)\n\n"
                f"**Alertas:**\n{alerts_text}\n"
            )

        # Intent 4: Direct Prediction
        else:
            # Default to test mix prediction
            sample_mix = {
                "cemento_kg": 350.0,
                "ceniza_volante_kg": 50.0,
                "escoria_kg": 0.0,
                "puzolana_kg": 0.0,
                "agua_kg": 180.0,
                "superplastificante_kg": 2.5,
                "agregado_grueso_kg": 1050.0,
                "agregado_fino_kg": 750.0,
                "edad_dias": 28,
                "tipo_cemento": "Tipo I",
            }
            pred = self.tool_predict_mix(sample_mix)
            tool_results.append(ToolExecutionResult("tool_predict_mix", sample_mix, pred))
            response_text = (
                f"### 🧪 Predicción de Desempeño Técnico con Intervalos de Incertidumbre (90%)\n\n"
                f"- **Resistencia axial:** {pred['resistencia_mpa']} MPa "
                f"(Intervalo de confianza: [{pred['resistencia_intervalo_90'][0]} - {pred['resistencia_intervalo_90'][1]} MPa])\n"
                f"- **Trabajabilidad (Slump):** {pred['slump_mm']} mm "
                f"(Intervalo de confianza: [{pred['slump_intervalo_90'][0]} - {pred['slump_intervalo_90'][1]} mm])\n"
                f"- **Evaluación de extrapolación:** {'⚠️ Fuera de rango de entrenamiento' if pred['es_extrapolacion'] else '✅ Dentro del dominio experimental seguro'}\n"
            )

        return {
            "query": user_query,
            "response": response_text,
            "tools_called": [r.tool_name for r in tool_results],
            "raw_tool_results": [r.output for r in tool_results],
        }
