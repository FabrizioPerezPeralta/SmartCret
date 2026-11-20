"""Multi-objective NSGA-II Concrete Mix Optimizer (Minimizing Cost & Embodied CO2)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from smartcret.models.predictor import ConcretePredictor, PredictionOutput


@dataclass
class OptimizationRequest:
    fc_target_mpa: float
    slump_target_mm: float
    age_days: int = 28
    exposure_class: str = "F0_normal"
    slump_tolerance_mm: float = 30.0
    cement_type: str = "Tipo I"
    materials_override: Optional[Dict[str, Any]] = None


@dataclass
class MixProposal:
    cemento_kg: float
    ceniza_volante_kg: float
    escoria_kg: float
    puzolana_kg: float
    agua_kg: float
    superplastificante_kg: float
    agregado_grueso_kg: float
    agregado_fino_kg: float
    relacion_ac: float
    tipo_cemento: str
    edad_dias: int
    costo_soles_m3: float
    co2_kg_m3: float
    fc_pred_mpa: float
    fc_lower_mpa: float
    fc_upper_mpa: float
    slump_pred_mm: float
    slump_lower_mm: float
    slump_upper_mm: float
    es_extrapolacion: bool
    perfil: str = "Pareto"  # 'Minimo Costo', 'Minimo CO2', 'Balanceado'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "perfil": self.perfil,
            "costo_s_m3": round(self.costo_soles_m3, 2),
            "co2_kg_m3": round(self.co2_kg_m3, 1),
            "resistencia_pred_mpa": round(self.fc_pred_mpa, 1),
            "resistencia_intervalo_90": [
                round(self.fc_lower_mpa, 1),
                round(self.fc_upper_mpa, 1),
            ],
            "slump_pred_mm": round(self.slump_pred_mm, 1),
            "slump_intervalo_90": [
                round(self.slump_lower_mm, 1),
                round(self.slump_upper_mm, 1),
            ],
            "dosificacion_kg_m3": {
                "cemento": round(self.cemento_kg, 1),
                "ceniza_volante": round(self.ceniza_volante_kg, 1),
                "escoria": round(self.escoria_kg, 1),
                "puzolana": round(self.puzolana_kg, 1),
                "agua": round(self.agua_kg, 1),
                "superplastificante": round(self.superplastificante_kg, 2),
                "agregado_grueso": round(self.agregado_grueso_kg, 1),
                "agregado_fino": round(self.agregado_fino_kg, 1),
            },
            "relacion_ac": round(self.relacion_ac, 3),
            "es_extrapolacion": self.es_extrapolacion,
        }


class MixOptimizer:
    """NSGA-II multi-objective optimizer balancing economic cost and carbon footprint."""

    def __init__(
        self,
        predictor_path: Union[str, Path] = "models/v1/predictor.pkl",
        norms_path: Union[str, Path] = "configs/norms.yaml",
        materials_path: Union[str, Path] = "configs/materials.yaml",
    ):
        self.predictor = ConcretePredictor.load(predictor_path)
        with open(norms_path, "r", encoding="utf-8") as f:
            self.norms = yaml.safe_load(f)
        with open(materials_path, "r", encoding="utf-8") as f:
            self.materials = yaml.safe_load(f)

    def _get_material_props(self, override: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, bool]]:
        mat_cfg = self.materials["materials"].copy()
        if override:
            for k, v in override.items():
                if k in mat_cfg:
                    mat_cfg[k].update(v)

        costs = {k: v["cost_per_unit"] for k, v in mat_cfg.items()}
        co2_factors = {k: v["co2_factor_kg"] for k, v in mat_cfg.items()}
        densities = {k: v["density_kg_m3"] for k, v in mat_cfg.items()}
        available = {k: v.get("available", True) for k, v in mat_cfg.items()}
        return costs, co2_factors, densities, available

    def _non_dominated_sort(self, F: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting for multi-objective optimization (minimization)."""
        n_points = F.shape[0]
        domination_count = np.zeros(n_points, dtype=int)
        dominated_indices: List[List[int]] = [[] for _ in range(n_points)]
        fronts: List[List[int]] = [[]]

        for p in range(n_points):
            for q in range(p + 1, n_points):
                # Check dominance (p dominates q if all <= and at least one <)
                p_dominates_q = (F[p, 0] <= F[q, 0] and F[p, 1] <= F[q, 1]) and (
                    F[p, 0] < F[q, 0] or F[p, 1] < F[q, 1]
                )
                q_dominates_p = (F[q, 0] <= F[p, 0] and F[q, 1] <= F[p, 1]) and (
                    F[q, 0] < F[p, 0] or F[q, 1] < F[p, 1]
                )

                if p_dominates_q:
                    dominated_indices[p].append(q)
                    domination_count[q] += 1
                elif q_dominates_p:
                    dominated_indices[q].append(p)
                    domination_count[p] += 1

            if domination_count[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_indices[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    def optimize(
        self,
        request: OptimizationRequest,
        population_size: int = 150,
        n_generations: int = 25,
        random_seed: int = 42,
    ) -> List[MixProposal]:
        """Runs NSGA-II to find Pareto front of concrete mixes for given requirements."""
        rng = np.random.default_rng(random_seed)
        costs, co2_factors, densities, available = self._get_material_props(
            request.materials_override
        )

        exposure = self.norms.get("exposure_classes", {}).get(request.exposure_class, {})
        max_norm_ac = exposure.get("max_relacion_ac", self.norms["general_limits"]["max_relacion_ac"])
        min_cemento = self.norms["general_limits"]["min_cemento_kg"]
        air_pct = self.norms["general_limits"].get("air_content_pct", 2.0) / 100.0

        # Decision variables: [total_binder, fly_ash_pct, slag_pct, pozzolan_pct, w_c, sp_dosage, sand_ratio]
        # Bounds based on norms and materials availability
        max_fa = self.norms["replacement_limits"]["max_fly_ash_pct"] if available["ceniza_volante"] else 0.0
        max_slag = self.norms["replacement_limits"]["max_slag_pct"] if available["escoria"] else 0.0
        max_pozz = self.norms["replacement_limits"]["max_pozzolan_pct"] if available["puzolana"] else 0.0

        low_bounds = np.array([280.0, 0.0, 0.0, 0.0, 0.35, 0.0, 0.38])
        up_bounds = np.array([480.0, max_fa, max_slag, max_pozz, min(max_norm_ac, 0.60), 0.020, 0.48])

        # Initialize population
        pop = rng.uniform(low_bounds, up_bounds, size=(population_size, len(low_bounds)))

        def evaluate_pop(population: np.ndarray) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[PredictionOutput]]:
            n = len(population)
            total_binder = population[:, 0]
            fa_pct = population[:, 1]
            slag_pct = population[:, 2]
            pozz_pct = population[:, 3]
            w_c = population[:, 4]
            sp_ratio = population[:, 5]
            sand_ratio = population[:, 6]

            # Normalize replacement so total SCM <= 0.50
            total_scm_pct = fa_pct + slag_pct + pozz_pct
            excess = np.maximum(total_scm_pct - 0.50, 0.0)
            scaling = np.where(total_scm_pct > 0.50, 0.50 / np.maximum(total_scm_pct, 1e-6), 1.0)
            fa_pct = fa_pct * scaling
            slag_pct = slag_pct * scaling
            pozz_pct = pozz_pct * scaling

            ceniza = total_binder * fa_pct
            escoria_m = total_binder * slag_pct
            puzolana_m = total_binder * pozz_pct
            cemento = total_binder - (ceniza + escoria_m + puzolana_m)

            # Ensure cemento >= min_cemento
            cemento = np.maximum(cemento, min_cemento)
            total_binder = cemento + ceniza + escoria_m + puzolana_m

            agua = total_binder * w_c
            sp_kg = total_binder * sp_ratio

            # Aggregate volume calculation (volumetric balance = 1.0 m³)
            v_paste = (
                cemento / densities["cemento"]
                + ceniza / densities["ceniza_volante"]
                + escoria_m / densities["escoria"]
                + puzolana_m / densities["puzolana"]
                + agua / densities["agua"]
                + sp_kg / densities["superplastificante"]
                + air_pct
            )
            v_agg = np.maximum(1.0 - v_paste, 0.50)
            agg_fino = v_agg * sand_ratio * densities["agregado_fino"]
            agg_grueso = v_agg * (1.0 - sand_ratio) * densities["agregado_grueso"]

            df_eval = pd.DataFrame(
                {
                    "cemento_kg": cemento,
                    "ceniza_volante_kg": ceniza,
                    "escoria_kg": escoria_m,
                    "puzolana_kg": puzolana_m,
                    "agua_kg": agua,
                    "superplastificante_kg": sp_kg,
                    "agregado_grueso_kg": agg_grueso,
                    "agregado_fino_kg": agg_fino,
                    "edad_dias": request.age_days,
                    "tipo_cemento": request.cement_type,
                    "relacion_ac": w_c,
                }
            )

            # Predict performance with uncertainty
            preds = self.predictor.predict(df_eval)

            # Economic cost (S/ / m³)
            cost = (
                cemento * costs["cemento"]
                + ceniza * costs["ceniza_volante"]
                + escoria_m * costs["escoria"]
                + puzolana_m * costs["puzolana"]
                + agua * costs["agua"]
                + sp_kg * costs["superplastificante"]
                + agg_grueso * costs["agregado_grueso"]
                + agg_fino * costs["agregado_fino"]
            )

            # Embodied CO2 (kg CO₂ / m³)
            co2 = (
                cemento * co2_factors["cemento"]
                + ceniza * co2_factors["ceniza_volante"]
                + escoria_m * co2_factors["escoria"]
                + puzolana_m * co2_factors["puzolana"]
                + agua * co2_factors["agua"]
                + sp_kg * co2_factors["superplastificante"]
                + agg_grueso * co2_factors["agregado_grueso"]
                + agg_fino * co2_factors["agregado_fino"]
            )

            # Constraint violations:
            # 1. fc lower bound (with safety margin) >= fc_target
            # 2. Slump within tolerance
            # 3. max a/c ratio
            fc_lowers = np.array([p.resistencia_lower for p in preds])
            slump_preds = np.array([p.slump_mm for p in preds])

            penalty_fc = np.maximum(request.fc_target_mpa - fc_lowers, 0.0)
            penalty_slump = np.maximum(
                np.abs(slump_preds - request.slump_target_mm) - request.slump_tolerance_mm, 0.0
            )
            penalty_ac = np.maximum(w_c - max_norm_ac, 0.0)

            total_penalty = penalty_fc * 25.0 + penalty_slump * 2.0 + penalty_ac * 500.0

            # Objectives with constraint penalty (Deb's penalty method)
            obj1 = cost + total_penalty * 2.0
            obj2 = co2 + total_penalty * 2.5
            F = np.column_stack([obj1, obj2])

            return F, total_penalty, df_eval, preds

        # Evolutionary loop (NSGA-II)
        for gen in range(n_generations):
            F, penalties, _, _ = evaluate_pop(pop)
            fronts = self._non_dominated_sort(F)

            # Selection via tournament
            selected_indices = []
            for _ in range(population_size):
                i1, i2 = rng.integers(0, population_size, size=2)
                # Compare ranks
                selected_indices.append(i1 if penalties[i1] < penalties[i2] else i2)

            parents = pop[selected_indices]

            # SBX crossover
            offspring = parents.copy()
            for i in range(0, population_size - 1, 2):
                if rng.random() < 0.9:
                    alpha = rng.uniform(0.1, 0.9)
                    offspring[i] = alpha * parents[i] + (1 - alpha) * parents[i + 1]
                    offspring[i + 1] = alpha * parents[i + 1] + (1 - alpha) * parents[i]

            # Polynomial mutation
            mutation_mask = rng.random(offspring.shape) < 0.15
            mutations = rng.normal(0, 0.05 * (up_bounds - low_bounds), size=offspring.shape)
            offspring = np.clip(
                np.where(mutation_mask, offspring + mutations, offspring),
                low_bounds,
                up_bounds,
            )

            # Combine population + offspring
            combined = np.vstack([pop, offspring])
            F_comb, pen_comb, _, _ = evaluate_pop(combined)
            fronts_comb = self._non_dominated_sort(F_comb)

            new_pop_indices = []
            for front in fronts_comb:
                if len(new_pop_indices) + len(front) <= population_size:
                    new_pop_indices.extend(front)
                else:
                    needed = population_size - len(new_pop_indices)
                    new_pop_indices.extend(front[:needed])
                    break

            pop = combined[new_pop_indices]

        # Final evaluation on resulting population
        F, penalties, df_final, preds = evaluate_pop(pop)
        costs_vec = (
            df_final["cemento_kg"] * costs["cemento"]
            + df_final["ceniza_volante_kg"] * costs["ceniza_volante"]
            + df_final["escoria_kg"] * costs["escoria"]
            + df_final["puzolana_kg"] * costs["puzolana"]
            + df_final["agua_kg"] * costs["agua"]
            + df_final["superplastificante_kg"] * costs["superplastificante"]
            + df_final["agregado_grueso_kg"] * costs["agregado_grueso"]
            + df_final["agregado_fino_kg"] * costs["agregado_fino"]
        ).values

        co2_vec = (
            df_final["cemento_kg"] * co2_factors["cemento"]
            + df_final["ceniza_volante_kg"] * co2_factors["ceniza_volante"]
            + df_final["escoria_kg"] * co2_factors["escoria"]
            + df_final["puzolana_kg"] * co2_factors["puzolana"]
            + df_final["agua_kg"] * co2_factors["agua"]
            + df_final["superplastificante_kg"] * co2_factors["superplastificante"]
            + df_final["agregado_grueso_kg"] * co2_factors["agregado_grueso"]
            + df_final["agregado_fino_kg"] * co2_factors["agregado_fino"]
        ).values

        # Filter strictly compliant mixes (penalty == 0)
        compliant_mask = penalties == 0.0
        if not compliant_mask.any():
            # If no mix strictly zero, take the 10 lowest penalty candidates
            best_idx = np.argsort(penalties)[:10]
        else:
            comp_indices = np.where(compliant_mask)[0]
            # Get Pareto front among compliant
            F_comp = np.column_stack([costs_vec[comp_indices], co2_vec[comp_indices]])
            fronts = self._non_dominated_sort(F_comp)
            best_idx = comp_indices[fronts[0]]

        # Build MixProposals
        proposals: List[MixProposal] = []
        for idx in best_idx:
            row = df_final.iloc[idx]
            pr = preds[idx]

            prop = MixProposal(
                cemento_kg=float(row["cemento_kg"]),
                ceniza_volante_kg=float(row["ceniza_volante_kg"]),
                escoria_kg=float(row["escoria_kg"]),
                puzolana_kg=float(row["puzolana_kg"]),
                agua_kg=float(row["agua_kg"]),
                superplastificante_kg=float(row["superplastificante_kg"]),
                agregado_grueso_kg=float(row["agregado_grueso_kg"]),
                agregado_fino_kg=float(row["agregado_fino_kg"]),
                relacion_ac=float(row["relacion_ac"]),
                tipo_cemento=str(row["tipo_cemento"]),
                edad_dias=int(row["edad_dias"]),
                costo_soles_m3=float(costs_vec[idx]),
                co2_kg_m3=float(co2_vec[idx]),
                fc_pred_mpa=float(pr.resistencia_mpa),
                fc_lower_mpa=float(pr.resistencia_lower),
                fc_upper_mpa=float(pr.resistencia_upper),
                slump_pred_mm=float(pr.slump_mm),
                slump_lower_mm=float(pr.slump_lower),
                slump_upper_mm=float(pr.slump_upper),
                es_extrapolacion=bool(pr.is_extrapolated),
                perfil="Pareto",
            )
            proposals.append(prop)

        # Remove duplicate proposals
        unique_props: List[MixProposal] = []
        for p in proposals:
            if not any(
                np.isclose(p.costo_soles_m3, u.costo_soles_m3, atol=1.0)
                and np.isclose(p.co2_kg_m3, u.co2_kg_m3, atol=2.0)
                for u in unique_props
            ):
                unique_props.append(p)

        # Label key representative archetypes: Min Cost, Min CO2, Balanced Knee
        if unique_props:
            unique_props.sort(key=lambda x: x.costo_soles_m3)
            unique_props[0].perfil = "Mínimo Costo"

            min_co2_prop = min(unique_props, key=lambda x: x.co2_kg_m3)
            min_co2_prop.perfil = "Mínimo CO₂"

            # Knee point (minimum distance to utopian point)
            min_c = min(p.costo_soles_m3 for p in unique_props)
            max_c = max(p.costo_soles_m3 for p in unique_props) + 1e-3
            min_e = min(p.co2_kg_m3 for p in unique_props)
            max_e = max(p.co2_kg_m3 for p in unique_props) + 1e-3

            def knee_dist(p: MixProposal):
                c_norm = (p.costo_soles_m3 - min_c) / (max_c - min_c)
                e_norm = (p.co2_kg_m3 - min_e) / (max_e - min_e)
                return np.sqrt(c_norm**2 + e_norm**2)

            knee_prop = min(unique_props, key=knee_dist)
            if knee_prop.perfil == "Pareto":
                knee_prop.perfil = "Ecoeficiente Balanceado"

        return unique_props
