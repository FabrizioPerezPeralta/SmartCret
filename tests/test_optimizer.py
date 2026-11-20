"""Unit tests for Phase 3: Multi-objective Concrete Mix Optimizer."""

import pytest
from smartcret.optimizer.optimizer import MixOptimizer, OptimizationRequest


@pytest.fixture(scope="module")
def optimizer():
    return MixOptimizer(
        predictor_path="models/v1/predictor.pkl",
        norms_path="configs/norms.yaml",
        materials_path="configs/materials.yaml",
    )


def test_optimizer_generates_compliant_mixes(optimizer):
    req = OptimizationRequest(
        fc_target_mpa=28.0,
        slump_target_mm=100.0,
        age_days=28,
        exposure_class="F0_normal",
        slump_tolerance_mm=35.0,
    )

    proposals = optimizer.optimize(req, population_size=60, n_generations=12, random_seed=42)

    assert len(proposals) > 0
    best_cost = proposals[0]
    assert best_cost.costo_soles_m3 > 0
    assert best_cost.co2_kg_m3 > 0
    # Must meet target strength (lower interval or predicted)
    assert best_cost.fc_pred_mpa >= 25.0
    # Material quantities must be valid physical amounts
    assert best_cost.cemento_kg >= 200.0
    assert 130.0 <= best_cost.agua_kg <= 250.0


def test_optimizer_respects_material_unavailability(optimizer):
    # Simulate shortage: no fly ash, no slag
    req = OptimizationRequest(
        fc_target_mpa=25.0,
        slump_target_mm=120.0,
        age_days=28,
        materials_override={
            "ceniza_volante": {"available": False},
            "escoria": {"available": False},
        },
    )

    proposals = optimizer.optimize(req, population_size=40, n_generations=8, random_seed=99)
    assert len(proposals) > 0
    for prop in proposals:
        assert prop.ceniza_volante_kg == 0.0
        assert prop.escoria_kg == 0.0
