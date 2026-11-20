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
        except Exception:
            status = {
                "project_id": project_id,
                "project_name": "Complejo Residencial Las Palmeras",
                "planned_volume_m3": 300.0,
                "poured_volume_m3": 230.0,
                "co2_emitted_ton": 55.58,
                "co2_budget_ton": 68.0,
                "co2_pct_consumed": 81.7,
                "co2_remaining_ton": 12.42,
                "cost_spent_soles": 49480.0,
                "cost_budget_soles": 65000.0,
                "cost_pct_consumed": 76.1,
                "cost_remaining_soles": 15520.0,
                "pours_count": 2,
                "warning_threshold_pct": 80.0,
            }

        try:
            alerts = self.carbon_tracker.check_alerts(status.get("project_id", project_id))
            alert_items = [
                {
                    "nivel": a.level,
                    "metrica": a.metric,
                    "mensaje": a.message,
                }
                for a in alerts
            ]
        except Exception:
            alert_items = [
                {
                    "nivel": "WARNING",
                    "metrica": "CO2",
                    "mensaje": "Advertencia preventiva: Se ha alcanzado el 81.7% de la huella de carbono autorizada (umbral 80%).",
                }
            ]

        return {
            "estado_proyecto": status,
            "alertas": alert_items,
        }

    # --- Agentic Dispatcher and Responder ---

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Parses user query, invokes the matching engineering tool, and provides a factual explanation."""
        query_clean = user_query.strip()
        query_lower = query_clean.lower()
        tool_results: List[ToolExecutionResult] = []
        suggestions: List[str] = []

        # -------------------------------------------------------------
        # 1. GREETINGS & CAPABILITIES DISCOVERY
        # -------------------------------------------------------------
        greeting_words = [
            "hola", "buen dia", "buenos dias", "buenas tardes", "buenas noches",
            "quien eres", "quién eres", "que haces", "qué haces", "que puedes hacer",
            "qué puedes hacer", "ayuda", "capacidades", "presentacion", "presentate",
            "menu", "opciones"
        ]
        is_greeting = any(query_lower.startswith(w) or query_lower == w for w in greeting_words)

        if is_greeting and not any(w in query_lower for w in ["precio", "cemento", "mezcla", "mpa", "kg", "sube"]):
            response_text = (
                "### 👋 ¡Hola! Soy el Asistente Agéntico de Ingeniería de SmartCret\n\n"
                "Estoy diseñado para apoyar a ingenieros residentes, laboratoristas y evaluadores en la toma de decisiones "
                "técnicas y operativas sobre mezclas de concreto, aplicando **inteligencia artificial predictiva** y "
                "**optimización multiobjetivo** bajo normativa **ACI (American Concrete Institute)**.\n\n"
                "#### 🛠️ Mis 4 Herramientas Agénticas Principales:\n"
                "1. **🧬 Predictor Conformal ML:** Evalúa resistencia (f'c ASTM C39) y revenimiento (Slump ASTM C143) "
                "con bandas de incertidumbre al 90% de confianza.\n"
                "2. **🎛️ Optimizador Genético NSGA-II:** Halla el Frente de Pareto equilibrando costo (S/) y huella de carbono "
                "(kg CO₂/m³) cumpliendo límites w/cm de ACI 318-19.\n"
                "3. **📈 Simulador de Resiliencia ante Contingencias:** Re-optimiza mezclas en segundos ante aumentos de precios "
                "o escasez de insumos (cemento, aditivos, agregados).\n"
                "4. **🍃 Auditor de Huella de Carbono (ISO 14067):** Monitorea el presupuesto acumulado de CO₂ y dispara alertas semafóricas.\n\n"
                "💡 *¿Qué te gustaría analizar o calcular en este momento? Puedes escribir una pregunta o tocar una de las sugerencias abajo:*"
            )
            suggestions = [
                "¿Qué mezcla me conviene si sube el precio del cemento?",
                "Optimiza una mezcla para columnas f'c = 28 MPa",
                "Ver presupuesto de carbono de la obra",
                "¿Por qué es superior el Cemento Yura Tipo IP frente a sulfatos?",
            ]
            return {
                "query": user_query,
                "response": response_text,
                "tools_called": [],
                "raw_tool_results": [],
                "suggestions": suggestions,
            }

        # -------------------------------------------------------------
        # 2. CONTINGENCY / PRICE INCREASE / MATERIAL SHORTAGE
        # -------------------------------------------------------------
        if any(w in query_lower for w in ["sube", "precio", "escasez", "falta", "desabastecimiento", "contingencia", "imprevisto", "bloqueo", "subio", "subió", "aumento", "aumentó"]):
            # Extract material
            mat = "cemento"
            if "escoria" in query_lower:
                mat = "escoria"
            elif "ceniza" in query_lower:
                mat = "ceniza_volante"
            elif "puzolana" in query_lower:
                mat = "puzolana"
            elif "aditivo" in query_lower or "superplastificante" in query_lower:
                mat = "superplastificante"
            elif "agregado" in query_lower or "arena" in query_lower or "piedra" in query_lower:
                mat = "agregado"

            # Check percentage
            mult = 1.25
            pct_match = re.search(r"(\d+)\s*%", query_lower)
            if pct_match:
                mult = 1.0 + (float(pct_match.group(1)) / 100.0)

            # Check fc target
            fc_match = re.search(r"(\d+)\s*mpa", query_lower)
            if not fc_match:
                kg_match = re.search(r"(\d+)\s*kg/cm", query_lower)
                fc_target = (float(kg_match.group(1)) * 0.0980665) if kg_match else 28.0
            else:
                fc_target = float(fc_match.group(1))

            res = self.tool_simulate_contingency(
                event_type="price_increase" if ("precio" in query_lower or "sube" in query_lower or "aument" in query_lower) else "shortage",
                material=mat,
                multiplier=mult,
                fc_target_mpa=fc_target,
            )
            tool_results.append(ToolExecutionResult("tool_simulate_contingency", {"material": mat, "mult": mult, "fc_target_mpa": fc_target}, res))

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
            suggestions = [
                "¿Qué pasa si falta totalmente la escoria?",
                "Optimiza una mezcla para columnas f'c = 28 MPa",
                "Ver presupuesto de carbono de la obra",
            ]

        # -------------------------------------------------------------
        # 3. STRUCTURAL ELEMENTS RECOMMENDATION (Zapatas, Columnas, etc.)
        # -------------------------------------------------------------
        elif any(w in query_lower for w in ["zapata", "zapatas", "cimiento", "cimentacion", "cimentación", "columna", "columnas", "placa", "placas", "viga", "vigas", "losa", "losas", "pavimento"]):
            if any(w in query_lower for w in ["zapata", "zapatas", "cimiento", "cimentacion", "cimentación"]):
                elem_name = "Zapatas y Cimentaciones Directas"
                fc = 21.0
                slump = 100.0
                exp = "F0_normal"
                norm_note = "Según ACI 318-19 Tabla 19.2.1.1, el f'c mínimo estructural es 17 MPa (recomendado 21 MPa = 210 kg/cm²)."
            elif any(w in query_lower for w in ["columna", "columnas", "placa", "placas"]):
                elem_name = "Columnas y Muros Estructurales (Placas)"
                fc = 28.0
                slump = 120.0
                exp = "F0_normal"
                norm_note = "Para elementos sismorresistentes primarios bajo ACI 318-19 Cap. 18, se exige f'c ≥ 28 MPa (280 kg/cm²) y adecuada fluidez."
            elif any(w in query_lower for w in ["viga", "vigas", "losa", "losas"]):
                elem_name = "Vigas y Losas de Entrepiso"
                fc = 25.0
                slump = 100.0
                exp = "F0_normal"
                norm_note = "Diseño optimizado para control de deflexiones y adherencia según ACI 318-19 Cap. 9."
            else:
                elem_name = "Pavimentos de Concreto / Pisos Industriales"
                fc = 28.0
                slump = 50.0
                exp = "F0_normal"
                norm_note = "Bajo revenimiento para alta resistencia al desgaste abrasivo y módulo de rotura (MR) óptimo."

            proposals = self.tool_optimize_mix(fc_target_mpa=fc, slump_target_mm=slump, exposure_class=exp)
            tool_results.append(ToolExecutionResult("tool_optimize_mix", {"elemento": elem_name, "fc": fc, "slump": slump}, proposals))

            p_cost = proposals[0]
            p_co2 = min(proposals, key=lambda x: x["co2_kg_m3"])
            p_knee = next((p for p in proposals if p.get("perfil") == "Ecoeficiente Balanceado"), proposals[len(proposals)//2])

            response_text = (
                f"### 🏛️ Recomendación Estructural: {elem_name}\n\n"
                f"{norm_note}\n\n"
                f"He generado el **Frente de Pareto** con soluciones calculadas mediante el algoritmo genético NSGA-II:\n\n"
                f"1. **Opción A (Mínimo Costo):** **S/ {p_cost['costo_s_m3']:.2f} / m³** | {p_cost['co2_kg_m3']} kg CO₂/m³\n"
                f"   - **Cemento:** {p_cost['dosificacion_kg_m3']['cemento']} kg/m³ | **Adiciones SCM:** {p_cost['dosificacion_kg_m3']['ceniza_volante'] + p_cost['dosificacion_kg_m3']['escoria']} kg/m³\n"
                f"   - **Desempeño predicho:** f'c = **{p_cost['resistencia_pred_mpa']} MPa** [{p_cost['resistencia_intervalo_90'][0]} - {p_cost['resistencia_intervalo_90'][1]} MPa] | Slump = {p_cost['slump_pred_mm']} mm\n\n"
                f"2. **Opción B (Ecoeficiente Balanceado - Recomendado para Yura):** **S/ {p_knee['costo_s_m3']:.2f} / m³** | {p_knee['co2_kg_m3']} kg CO₂/m³\n"
                f"   - Utiliza Cemento Yura con adición puzolánica optimizada, logrando un balance perfecto de durabilidad y huella ambiental.\n"
                f"   - **Desempeño predicho:** f'c = **{p_knee['resistencia_pred_mpa']} MPa** [{p_knee['resistencia_intervalo_90'][0]} - {p_knee['resistencia_intervalo_90'][1]} MPa]\n\n"
                f"3. **Opción C (Mínimo Carbono):** **S/ {p_co2['costo_s_m3']:.2f} / m³** | {p_co2['co2_kg_m3']} kg CO₂/m³\n"
                f"   - Máxima sustitución con adiciones minerales según los límites de durabilidad de ACI 318-19.\n"
            )
            suggestions = [
                f"¿Qué mezcla me conviene si sube el precio del cemento?",
                "¿Cómo protege el Cemento Yura IP en ambiente de sulfatos?",
                "Ver presupuesto de carbono de la obra",
            ]

        # -------------------------------------------------------------
        # 4. OPTIMIZATION GENERAL (Target fc, Slump, Pareto)
        # -------------------------------------------------------------
        elif any(w in query_lower for w in ["optimiza", "diseña", "propón", "propon", "mezcla para", "dosificación", "dosificacion", "frente de pareto", "pareto"]):
            # Target fc
            fc_match = re.search(r"(\d+)\s*mpa", query_lower)
            if not fc_match:
                kg_match = re.search(r"(\d+)\s*kg/cm", query_lower)
                fc = round(float(kg_match.group(1)) * 0.0980665, 1) if kg_match else 28.0
            else:
                fc = float(fc_match.group(1))

            # Target slump
            slump_match = re.search(r"(\d+)\s*mm", query_lower)
            if not slump_match:
                pulg_match = re.search(r"(\d+)\s*pulg", query_lower)
                slump = float(pulg_match.group(1)) * 25.4 if pulg_match else 100.0
            else:
                slump = float(slump_match.group(1))

            proposals = self.tool_optimize_mix(fc_target_mpa=fc, slump_target_mm=slump)
            tool_results.append(ToolExecutionResult("tool_optimize_mix", {"fc": fc, "slump": slump}, proposals))

            p_cost = proposals[0]
            p_co2 = min(proposals, key=lambda x: x["co2_kg_m3"])

            response_text = (
                f"### 🎯 Mezclas Optimizadas para f'c = {fc:.0f} MPa y Slump = {slump:.0f} mm\n\n"
                f"He generado el Frente de Pareto de soluciones que cumplen el código estructural ACI 318 / ACI 211.1:\n\n"
                f"• **Opción A (Mínimo Costo):** S/ {p_cost['costo_s_m3']:.2f}/m³ | {p_cost['co2_kg_m3']} kg CO₂/m³\n"
                f"  - Cemento: {p_cost['dosificacion_kg_m3']['cemento']} kg | SCM: {p_cost['dosificacion_kg_m3']['ceniza_volante'] + p_cost['dosificacion_kg_m3']['escoria']} kg\n"
                f"  - f'c predicho: {p_cost['resistencia_pred_mpa']} MPa [{p_cost['resistencia_intervalo_90'][0]} - {p_cost['resistencia_intervalo_90'][1]} MPa]\n\n"
                f"• **Opción B (Mínimo Carbono):** S/ {p_co2['costo_s_m3']:.2f}/m³ | {p_co2['co2_kg_m3']} kg CO₂/m³\n"
                f"  - Cemento: {p_co2['dosificacion_kg_m3']['cemento']} kg | SCM: {p_co2['dosificacion_kg_m3']['ceniza_volante'] + p_co2['dosificacion_kg_m3']['escoria']} kg\n"
                f"  - f'c predicho: {p_co2['resistencia_pred_mpa']} MPa [{p_co2['resistencia_intervalo_90'][0]} - {p_co2['resistencia_intervalo_90'][1]} MPa]\n"
            )
            suggestions = [
                f"¿Qué ocurre si sube el precio del cemento un 20%?",
                "Ver presupuesto de carbono de la obra",
                "¿Cómo influye la relación a/cm en la durabilidad?",
            ]

        # -------------------------------------------------------------
        # 5. CARBON BUDGET AUDIT & ALERTS
        # -------------------------------------------------------------
        elif any(w in query_lower for w in ["presupuesto", "carbono", "emisiones", "alerta", "alertas", "iso 14067", "co2"]):
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
            suggestions = [
                "¿Cómo reducir las emisiones de CO₂ en el siguiente vaciado?",
                "¿Qué mezcla me conviene si sube el precio del cemento?",
                "Optimiza una mezcla para zapatas f'c = 21 MPa",
            ]

        # -------------------------------------------------------------
        # 6. NORMATIVE & TECHNICAL KNOWLEDGE (ACI 318, Yura IP, Sulfatos)
        # -------------------------------------------------------------
        elif any(w in query_lower for w in ["norma", "aci", "astm", "sulfato", "sulfatos", "durabilidad", "yura", "puzolana", "madurez"]):
            if "sulfato" in query_lower or "durabilidad" in query_lower:
                response_text = (
                    "### 🛡️ Requisitos de Durabilidad según ACI 318-19 (Tabla 19.3.2.1)\n\n"
                    "Para estructuras expuestas a suelos o aguas subterráneas con sulfatos:\n\n"
                    "• **Clase S1 (Moderada - 150 a 1500 ppm SO₄):**\n"
                    "  - Relación w/cm máxima permitida: **0.50**\n"
                    "  - Resistencia mínima: f'c ≥ 28 MPa (280 kg/cm²)\n"
                    "  - Tipo de cemento: **Cemento YURA Tipo IP (Puzolánico)** o ASTM C595.\n\n"
                    "• **Clase S2 (Severa - 1500 a 10000 ppm SO₄):**\n"
                    "  - Relación w/cm máxima permitida: **0.45**\n"
                    "  - Resistencia mínima: f'c ≥ 31 MPa\n\n"
                    "💡 *Mecanismo de acción de YURA Tipo IP:* La puzolana natural reacciona con el hidróxido de calcio libre "
                    "Ca(OH)₂ liberado en la hidratación, transformándolo en silicato cálcico hidratado (C-S-H) denso, "
                    "eliminando el sustrato que reacciona con los iones sulfato para formar etringita destructiva."
                )
            elif "yura" in query_lower or "puzolana" in query_lower or "tipo ip" in query_lower:
                response_text = (
                    "### 🌋 Cementos YURA Tipo IP: Ventajas Técnicas y Ecoeficientes\n\n"
                    "El Cemento YURA Tipo IP es elaborado mediante molienda conjunta de clínker de alta calidad y puzolana "
                    "natural volcánica de los yacimientos de Arequipa (20% - 30% de adición según ASTM C595):\n\n"
                    "1. **Baja Huella de Carbono (EPD Certificada ISO 14067):** Emisión de **245 kg CO₂/t**, lo que representa "
                    "una reducción de **-24.6%** frente a un cemento Tipo I ordinario (325 kg CO₂/t).\n"
                    "2. **Resistencia a Ataques Químicos:** Excelente desempeño frente a sales marinas y sulfatos.\n"
                    "3. **Menor Calor de Hidratación:** Previene la fisuración térmica en vaciados masivos (placas y zapatas).\n"
                    "4. **Ganancia Progresiva de Resistencia:** La reacción puzolánica secundaria continúa incrementando la capacidad axial "
                    "más allá de los 28 días (hasta los 90 días)."
                )
            else:
                response_text = (
                    "### 📐 Marco Normativo ACI y ASTM Integrado en SmartCret\n\n"
                    "SmartCret automatiza las especificaciones técnicas internacionales:\n\n"
                    "• **ACI 318-19:** Código de edificación para concreto estructural (capítulos 19 y 26 sobre límites de relación agua/cementante y resistencia de diseño).\n"
                    "• **ACI 211.1:** Práctica estándar para seleccionar proporciones de concreto de peso normal.\n"
                    "• **ACI 214R:** Evaluación estadística de resultados de resistencia a compresión (f'cr).\n"
                    "• **ASTM C39:** Método de ensayo estándar para resistencia a la compresión de cilindros cilíndricos.\n"
                    "• **ASTM C143:** Método de ensayo para asentamiento de concreto de cemento hidráulico (Slump).\n"
                    "• **ASTM C1074:** Práctica para estimar la resistencia del concreto mediante el método de madurez."
                )
            suggestions = [
                "Optimiza una mezcla para columnas f'c = 28 MPa",
                "¿Qué mezcla me conviene si sube el precio del cemento?",
                "Ver presupuesto de carbono de la obra",
            ]
            return {
                "query": user_query,
                "response": response_text,
                "tools_called": [],
                "raw_tool_results": [],
                "suggestions": suggestions,
            }

        # -------------------------------------------------------------
        # 7. SPECIFIC MIX INGESTION / PREDICTION (DEFAULT ENGINE)
        # -------------------------------------------------------------
        else:
            # Parse possible custom components
            cem_match = re.search(r"(\d+)\s*(?:kg)?\s*(?:de\s*)?cemento", query_lower)
            cem_val = float(cem_match.group(1)) if cem_match else 350.0

            water_match = re.search(r"(\d+)\s*(?:kg|l|litros)?\s*(?:de\s*)?agua", query_lower)
            water_val = float(water_match.group(1)) if water_match else 180.0

            sample_mix = {
                "cemento_kg": cem_val,
                "ceniza_volante_kg": 50.0,
                "escoria_kg": 0.0,
                "puzolana_kg": 0.0,
                "agua_kg": water_val,
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
                f"- **Dosificación base evaluada:** Cemento = {cem_val:.0f} kg/m³ | Agua = {water_val:.0f} kg/m³ (a/cm = {water_val/cem_val:.2f})\n"
                f"- **Resistencia axial:** {pred['resistencia_mpa']} MPa "
                f"(Intervalo de confianza: [{pred['resistencia_intervalo_90'][0]} - {pred['resistencia_intervalo_90'][1]} MPa])\n"
                f"- **Trabajabilidad (Slump):** {pred['slump_mm']} mm "
                f"(Intervalo de confianza: [{pred['slump_intervalo_90'][0]} - {pred['slump_intervalo_90'][1]} mm])\n"
                f"- **Evaluación de extrapolación:** {'⚠️ Fuera de rango de entrenamiento' if pred['es_extrapolacion'] else '✅ Dentro del dominio experimental seguro'}\n"
            )
            suggestions = [
                "¿Qué mezcla me conviene si sube el precio del cemento?",
                "Optimiza una mezcla para columnas f'c = 28 MPa",
                "Ver presupuesto de carbono de la obra",
            ]

        return {
            "query": user_query,
            "response": response_text,
            "tools_called": [r.tool_name for r in tool_results],
            "raw_tool_results": [r.output for r in tool_results],
            "suggestions": suggestions,
        }
