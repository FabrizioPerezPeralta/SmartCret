"""Unit tests for Phase 4: Contingency simulation, carbon budgeting, and scenario management."""

import pytest
from pathlib import Path
from smartcret.optimizer.optimizer import MixOptimizer, MixProposal, OptimizationRequest
from smartcret.simulator.carbon_budget import CarbonBudgetTracker, ProjectCarbonBudget
from smartcret.simulator.scenarios import ScenarioManager, ConcreteScenario
from smartcret.simulator.simulator import ContingencySimulator, ContingencyEvent


@pytest.fixture
def temp_db(tmp_path):
    return str(tmp_path / "test_smartcret.db")


def test_scenario_save_reload_and_compare(temp_db):
    mgr = ScenarioManager(db_path=temp_db)

    mix1 = MixProposal(
        cemento_kg=350.0,
        ceniza_volante_kg=50.0,
        escoria_kg=0.0,
        puzolana_kg=0.0,
        agua_kg=175.0,
        superplastificante_kg=2.5,
        agregado_grueso_kg=1050.0,
        agregado_fino_kg=750.0,
        relacion_ac=0.438,
        tipo_cemento="Tipo I",
        edad_dias=28,
        costo_soles_m3=215.50,
        co2_kg_m3=285.2,
        fc_pred_mpa=32.4,
        fc_lower_mpa=29.1,
        fc_upper_mpa=35.7,
        slump_pred_mm=105.0,
        slump_lower_mm=85.0,
        slump_upper_mm=125.0,
        es_extrapolacion=False,
        perfil="Mínimo Costo",
    )

    mix2 = MixProposal(
        cemento_kg=280.0,
        ceniza_volante_kg=70.0,
        escoria_kg=50.0,
        puzolana_kg=0.0,
        agua_kg=165.0,
        superplastificante_kg=3.2,
        agregado_grueso_kg=1020.0,
        agregado_fino_kg=780.0,
        relacion_ac=0.413,
        tipo_cemento="Tipo I",
        edad_dias=28,
        costo_soles_m3=204.80,
        co2_kg_m3=232.0,
        fc_pred_mpa=31.2,
        fc_lower_mpa=28.0,
        fc_upper_mpa=34.4,
        slump_pred_mm=95.0,
        slump_lower_mm=75.0,
        slump_upper_mm=115.0,
        es_extrapolacion=False,
        perfil="Mínimo CO₂",
    )

    s1 = ConcreteScenario(
        scenario_id="scen_costo",
        name="Mezcla Estándar Económica",
        project_name="Edificio Las Palmas",
        requirements={"fc": 28.0, "slump": 100.0},
        mix=mix1,
    )
    s2 = ConcreteScenario(
        scenario_id="scen_verde",
        name="Mezcla Ecoeficiente con Escoria",
        project_name="Edificio Las Palmas",
        requirements={"fc": 28.0, "slump": 100.0},
        mix=mix2,
    )

    mgr.save_scenario(s1)
    mgr.save_scenario(s2)

    loaded_list = mgr.list_scenarios()
    assert len(loaded_list) == 2

    reloaded_s1 = mgr.get_scenario("scen_costo")
    assert reloaded_s1.name == "Mezcla Estándar Económica"
    assert reloaded_s1.mix.costo_soles_m3 == 215.50

    comp_df = mgr.compare_scenarios(["scen_costo", "scen_verde"])
    assert comp_df.shape[1] == 2
    assert "Costo (S/ / m³)" in comp_df.index


def test_carbon_budget_alerts(temp_db):
    tracker = CarbonBudgetTracker(db_path=temp_db)
    project = ProjectCarbonBudget(
        project_id="PROJ-001",
        project_name="Puente Metropolitano",
        planned_volume_m3=100.0,
        co2_budget_ton=25.0,       # 25 tons
        cost_budget_soles=25000.0, # S/ 25,000
        warning_threshold_pct=80.0,
    )
    tracker.create_project(project)

    # Pour 1: 50 m³ -> 12.5 ton CO2 (50% budget) -> No alerts
    alerts1 = tracker.log_pour(
        project_id="PROJ-001",
        volume_m3=50.0,
        unit_co2_kg_m3=250.0,  # 0.25 ton / m³ -> 12.5 ton
        unit_cost_s_m3=220.0,
    )
    assert len(alerts1) == 0

    # Pour 2: 35 m³ -> total 85 m³, 21.25 ton CO2 (85% budget) -> WARNING alert!
    alerts2 = tracker.log_pour(
        project_id="PROJ-001",
        volume_m3=35.0,
        unit_co2_kg_m3=250.0,
        unit_cost_s_m3=220.0,
    )
    assert any(a.level == "WARNING" and a.metric == "CO2" for a in alerts2)

    # Pour 3: 20 m³ -> total 105 m³, 26.25 ton CO2 (105% budget) -> CRITICAL alert!
    alerts3 = tracker.log_pour(
        project_id="PROJ-001",
        volume_m3=20.0,
        unit_co2_kg_m3=250.0,
        unit_cost_s_m3=220.0,
    )
    assert any(a.level == "CRITICAL" and a.metric == "CO2" for a in alerts3)


def test_pdf_report_generation(temp_db, tmp_path):
    mgr = ScenarioManager(db_path=temp_db)
    mix = MixProposal(
        cemento_kg=320.0,
        ceniza_volante_kg=40.0,
        escoria_kg=0.0,
        puzolana_kg=0.0,
        agua_kg=170.0,
        superplastificante_kg=2.0,
        agregado_grueso_kg=1050.0,
        agregado_fino_kg=760.0,
        relacion_ac=0.472,
        tipo_cemento="Tipo I",
        edad_dias=28,
        costo_soles_m3=210.0,
        co2_kg_m3=265.0,
        fc_pred_mpa=30.0,
        fc_lower_mpa=27.0,
        fc_upper_mpa=33.0,
        slump_pred_mm=100.0,
        slump_lower_mm=80.0,
        slump_upper_mm=120.0,
        es_extrapolacion=False,
    )
    s = ConcreteScenario(
        scenario_id="scen_pdf",
        name="Escenario PDF Demo",
        project_name="Proyecto Demo",
        requirements={"fc": 28.0, "slump": 100.0},
        mix=mix,
    )
    mgr.save_scenario(s)

    pdf_path = tmp_path / "reporte_test.pdf"
    res = mgr.export_pdf_report(["scen_pdf"], pdf_path)
    assert Path(res).exists()
    assert Path(res).stat().st_size > 1000
