"""Unforeseen event simulator: price volatility, material shortages, and real-time re-optimization."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import yaml

from smartcret.optimizer.optimizer import MixOptimizer, MixProposal, OptimizationRequest


@dataclass
class ContingencyEvent:
    name: str
    price_multipliers: Dict[str, float] = field(default_factory=dict)  # e.g. {'cemento': 1.25}
    unavailable_materials: List[str] = field(default_factory=list)      # e.g. ['escoria', 'ceniza_volante']
    description: str = ""


@dataclass
class ContingencyImpact:
    cost_original: float
    cost_new: float
    cost_delta: float
    cost_delta_pct: float
    co2_original: float
    co2_new: float
    co2_delta: float
    co2_delta_pct: float
    material_substitutions: Dict[str, float]  # delta in kg/m³
    engineering_summary: str


class ContingencySimulator:
    """Simulates economic or supply chain disruption scenarios and assesses re-optimization options."""

    def __init__(self, optimizer: Optional[MixOptimizer] = None):
        self.optimizer = optimizer or MixOptimizer()

    def simulate(
        self,
        base_request: OptimizationRequest,
        event: ContingencyEvent,
        random_seed: int = 42,
    ) -> Tuple[MixProposal, MixProposal, ContingencyImpact]:
        """Runs baseline optimization and contingency re-optimization, computing impact metrics."""
        # 1. Baseline optimization
        base_proposals = self.optimizer.optimize(
            base_request, population_size=60, n_generations=12, random_seed=random_seed
        )
        if not base_proposals:
            raise RuntimeError("No se encontró solución factible para la solicitud base.")
        base_mix = base_proposals[0]

        # 2. Build material override dictionary
        override = {}
        for mat, mult in event.price_multipliers.items():
            base_cost = self.optimizer.materials["materials"].get(mat, {}).get("cost_per_unit", 1.0)
            override[mat] = {"cost_per_unit": base_cost * mult}

        for mat in event.unavailable_materials:
            if mat not in override:
                override[mat] = {}
            override[mat]["available"] = False

        # 3. Contingency optimization
        contingency_request = OptimizationRequest(
            fc_target_mpa=base_request.fc_target_mpa,
            slump_target_mm=base_request.slump_target_mm,
            age_days=base_request.age_days,
            exposure_class=base_request.exposure_class,
            slump_tolerance_mm=base_request.slump_tolerance_mm,
            cement_type=base_request.cement_type,
            materials_override=override,
        )

        cont_proposals = self.optimizer.optimize(
            contingency_request, population_size=60, n_generations=12, random_seed=random_seed
        )
        if not cont_proposals:
            raise RuntimeError(
                f"No fue posible encontrar una mezcla factible bajo el evento: {event.name}"
            )
        new_mix = cont_proposals[0]

        # 4. Impact calculation
        cost_delta = new_mix.costo_soles_m3 - base_mix.costo_soles_m3
        cost_delta_pct = (cost_delta / base_mix.costo_soles_m3) * 100.0

        co2_delta = new_mix.co2_kg_m3 - base_mix.co2_kg_m3
        co2_delta_pct = (co2_delta / base_mix.co2_kg_m3) * 100.0

        subs = {
            "cemento_kg": round(new_mix.cemento_kg - base_mix.cemento_kg, 1),
            "ceniza_volante_kg": round(new_mix.ceniza_volante_kg - base_mix.ceniza_volante_kg, 1),
            "escoria_kg": round(new_mix.escoria_kg - base_mix.escoria_kg, 1),
            "puzolana_kg": round(new_mix.puzolana_kg - base_mix.puzolana_kg, 1),
            "agua_kg": round(new_mix.agua_kg - base_mix.agua_kg, 1),
            "superplastificante_kg": round(new_mix.superplastificante_kg - base_mix.superplastificante_kg, 2),
        }

        # Build descriptive engineering synthesis
        lines = []
        lines.append(f"Evento simulado: '{event.name}'.")
        if event.price_multipliers:
            chg = ", ".join([f"{k} ({v:+.0%})" for k, v in event.price_multipliers.items()])
            lines.append(f"Variación de precios: {chg}.")
        if event.unavailable_materials:
            lines.append(f"Materiales restringidos por desabastecimiento: {', '.join(event.unavailable_materials)}.")

        lines.append(
            f"Impacto económico: S/ {cost_delta:+.2f}/m³ ({cost_delta_pct:+.1f}%). "
            f"Impacto huella de carbono: {co2_delta:+.1f} kg CO₂/m³ ({co2_delta_pct:+.1f}%)."
        )

        impact = ContingencyImpact(
            cost_original=round(base_mix.costo_soles_m3, 2),
            cost_new=round(new_mix.costo_soles_m3, 2),
            cost_delta=round(cost_delta, 2),
            cost_delta_pct=round(cost_delta_pct, 1),
            co2_original=round(base_mix.co2_kg_m3, 1),
            co2_new=round(new_mix.co2_kg_m3, 1),
            co2_delta=round(co2_delta, 1),
            co2_delta_pct=round(co2_delta_pct, 1),
            material_substitutions=subs,
            engineering_summary=" ".join(lines),
        )

        return base_mix, new_mix, impact
