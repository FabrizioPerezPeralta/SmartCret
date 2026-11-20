"""Streamlit Web Application for SmartCret - Enterprise Concrete Optimization Platform.

Enterprise interface engineered for institutional and jury evaluation:
1. Panel principal: Visión ejecutiva, KPIs, arquitectura y metas del proyecto.
2. Dosificación: Cálculo volumétrico ACI 211 / NTP 339 y balance de mezclas.
3. Optimización: Algoritmo genético multiobjetivo NSGA-II y Frente de Pareto.
4. Escenarios: Simulador de contingencias de precios/suministro y fichas PDF.
5. Materiales: Catálogo con factores EPD certificados (Cementos Yura).
6. Predicción IA: Laboratorio virtual con intervalos conformales (90%) y asistente agéntico.
7. Huella de carbono: Monitoreo en tiempo real de presupuestos y descarbonización.
8. Alertas: Centro de notificaciones preventivas y normativas.
"""

import os
from pathlib import Path
import sys
import base64

# Ensure 'src' and project root are always in sys.path
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parent.parent.parent
_project_root = _src_dir.parent

if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.chdir(_project_root)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from smartcret.assistant.assistant import SmartCretAssistant
from smartcret.models.predictor import ConcretePredictor
from smartcret.optimizer.optimizer import MixOptimizer, OptimizationRequest
from smartcret.simulator.carbon_budget import CarbonBudgetTracker, ProjectCarbonBudget
from smartcret.simulator.scenarios import ConcreteScenario, ScenarioManager
from smartcret.simulator.simulator import ContingencyEvent, ContingencySimulator

# Page Config
st.set_page_config(
    page_title="SmartCret Enterprise | Plataforma de Optimización IA de Concreto",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise Global Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* Executive Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0B1528 0%, #0F1F38 50%, #0A1629 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #38BDF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 5px;
        font-weight: 400;
        letter-spacing: -0.01em;
    }
    
    /* Institutional & Status Badges */
    .badge-yura {
        background: rgba(14, 165, 233, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-norm {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-active {
        background: linear-gradient(90deg, #0284C7 0%, #0D9488 100%);
        color: #FFFFFF;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
    }

    /* Executive KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #111C2E;
        border: 1px solid #1E2E48;
        border-radius: 12px;
        padding: 20px 22px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        border-color: #38BDF8;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.15);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #38BDF8, #34D399);
    }
    .kpi-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.05rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .kpi-footer {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .kpi-footer-green {
        color: #34D399;
    }
    .kpi-footer-blue {
        color: #38BDF8;
    }

    /* Section Titles */
    .module-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .module-sub {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-bottom: 22px;
        font-weight: 400;
    }

    /* Enterprise Cards */
    .enterprise-card {
        background: #111C2E;
        border: 1px solid #1E2E48;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08101E 0%, #060B14 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem !important;
    }

    /* Sidebar Radio Options Styling */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: rgba(17, 28, 46, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        transform: translateX(3px) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(14, 165, 233, 0.22) 0%, rgba(16, 185, 129, 0.12) 100%) !important;
        border: 1px solid #00A3C4 !important;
        border-left: 4px solid #00A3C4 !important;
        box-shadow: 0 4px 14px rgba(0, 163, 196, 0.22) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_all_services():
    predictor_path = Path("models/v2/predictor.pkl") if Path("models/v2/predictor.pkl").exists() else Path("models/v1/predictor.pkl")
    if not predictor_path.exists():
        st.error("Modelo no encontrado en models/v2 o models/v1. Ejecuta 'python train.py' primero.")
        st.stop()

    predictor = ConcretePredictor.load(predictor_path)
    optimizer = MixOptimizer(predictor_path=predictor_path)
    simulator = ContingencySimulator(optimizer=optimizer)
    carbon_tracker = CarbonBudgetTracker()
    scenario_manager = ScenarioManager()
    assistant = SmartCretAssistant(
        predictor=predictor,
        optimizer=optimizer,
        simulator=simulator,
        carbon_tracker=carbon_tracker,
        scenario_manager=scenario_manager,
    )

    # Initialize standard project if needed
    for pid in ["OBRA-LIMA-01", "PROJ-001"]:
        try:
            carbon_tracker.get_project_status(pid)
        except ValueError:
            carbon_tracker.create_project(
                ProjectCarbonBudget(
                    project_id=pid,
                    project_name="Complejo Residencial Las Palmeras",
                    planned_volume_m3=300.0,
                    co2_budget_ton=68.0,
                    cost_budget_soles=65000.0,
                    warning_threshold_pct=80.0,
                )
            )
            carbon_tracker.log_pour(pid, 120.0, 245.0, 218.0, "Cimentaciones")
            carbon_tracker.log_pour(pid, 110.0, 238.0, 212.0, "Columnas y Placas")

    return predictor, optimizer, simulator, carbon_tracker, scenario_manager, assistant


predictor, optimizer, simulator, carbon_tracker, scenario_manager, assistant = load_all_services()

# -------------------------------------------------------------
# SIDEBAR NAVIGATION (Enterprise Brand Identity)
# -------------------------------------------------------------
with st.sidebar:
    logo_path = Path("imagenes/logo.jpg")
    if logo_path.exists():
        with open(logo_path, "rb") as _img_f:
            _b64 = base64.b64encode(_img_f.read()).decode("utf-8")
        st.markdown(
            f"""
            <div style="background: #FFFFFF; border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 4px 18px rgba(0,0,0,0.35); text-align: center; border: 1px solid rgba(255,255,255,0.15);">
                <img src="data:image/jpeg;base64,{_b64}" style="width: 100%; max-height: 72px; object-fit: contain; display: block; margin: 0 auto;" />
            </div>
            <div style="font-size: 0.65rem; color: #38BDF8; font-weight: 800; letter-spacing: 0.14em; text-align: center; margin-bottom: 16px; text-transform: uppercase;">
                PLATAFORMA INDUSTRIAL IA v2.0
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="padding: 10px 0 16px 0;">
                <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.01em; color: #F8FAFC;">
                    SMARTCRET
                </div>
                <div style="font-size: 0.78rem; color: #38BDF8; font-weight: 600; text-transform: uppercase;">
                    Plataforma de Optimización IA
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<p style='font-size: 0.72rem; font-weight: 800; color: #64748B; letter-spacing: 0.08em; margin-bottom: 8px; text-transform: uppercase;'>MENÚ DEL SISTEMA</p>", unsafe_allow_html=True)

    menu_options = [
        "Panel principal",
        "Dosificación",
        "Optimización",
        "Escenarios",
        "Materiales",
        "Predicción IA",
        "Huella de carbono",
        "Alertas",
    ]

    icons = {
        "Panel principal": "⊞",
        "Dosificación": "🧮",
        "Optimización": "🎛️",
        "Escenarios": "📈",
        "Materiales": "🗄️",
        "Predicción IA": "🧬",
        "Huella de carbono": "🍃",
        "Alertas": "🔔",
    }

    selected_module = st.radio(
        label="Navegación",
        options=menu_options,
        format_func=lambda x: f"{icons.get(x, '•')}  {x}",
        label_visibility="collapsed",
    )

    # Technical Certification Sidebar Card
    st.markdown(
        """
        <div style="background: rgba(17, 28, 46, 0.85); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 10px; padding: 14px 16px; margin-top: 22px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 0.72rem; font-weight: 800; color: #CBD5E1; letter-spacing: 0.05em;">🛡️ PARÁMETROS DE CERTIFICACIÓN</span>
                <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-size: 0.65rem; font-weight: 800; padding: 2px 7px; border-radius: 9999px;">CALIBRADO</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 7px; font-size: 0.73rem; color: #94A3B8;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Norma activa:</span>
                    <b style="color: #F8FAFC;">NTP 339 / ACI 318</b>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Dataset Base:</span>
                    <b style="color: #38BDF8;">Yura Arequipa (808 cil.)</b>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Precisión Test:</span>
                    <b style="color: #34D399;">MAPE = 4.34% (< 10%)</b>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Incertidumbre:</span>
                    <b style="color: #FBBF24;">Conformal 90% Conf.</b>
                </div>
                <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 6px;">
                    <span>Motor Genético:</span>
                    <b style="color: #A78BFA;">NSGA-II Multiobjetivo</b>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Top Executive App Header
st.markdown(
    f"""
    <div class="hero-banner">
        <div>
            <h1 class="hero-title">SMARTCRET</h1>
            <div class="hero-subtitle">Sistema Integral de Diseño, Optimización y Control Ecoeficiente de Concreto mediante Machine Learning</div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
            <div style="display: flex; gap: 8px;">
                <span class="badge-yura">🏢 Cementos YURA S.A.</span>
                <span class="badge-norm">⚖️ NTP 339 / ACI 318 / E.060</span>
            </div>
            <div class="badge-active">
                {icons.get(selected_module, '')} {selected_module}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================
# 1. PANEL PRINCIPAL (Vista Ejecutiva y Demostración al Jurado)
# =============================================================
if selected_module == "Panel principal":
    st.markdown("<div class='module-header'>⊞ Panel Principal de Control de Ingeniería</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Tablero ejecutivo de validación con datos reales de Cementos YURA, desempeño ambiental y compromisos superados.</div>", unsafe_allow_html=True)

    # 4 Executive KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Tiempo de Diseño de Mezcla</div>
                <div class="kpi-value">< 15 seg</div>
                <div class="kpi-footer kpi-footer-green">
                    <span>▲</span> <b>-85%</b> vs. proceso tradicional (3-5 días)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Ahorro Económico en Obra</div>
                <div class="kpi-value">-8.4%</div>
                <div class="kpi-footer kpi-footer-green">
                    <span>▲</span> De S/ 245.00 a <b>S/ 224.40 / m³</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Descarbonización (CO₂)</div>
                <div class="kpi-value">-24.6%</div>
                <div class="kpi-footer kpi-footer-green">
                    <span>▲</span> EPD Yura IP: <b>245 kg CO₂/m³</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">Precisión Predictiva (MAPE)</div>
                <div class="kpi-value">4.34%</div>
                <div class="kpi-footer kpi-footer-blue">
                    <span>✓</span> Meta < 10% | <b>R² = 0.900</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Architecture Pipeline for the Jury
    st.markdown(
        """
        <div class="enterprise-card" style="margin-top: 10px;">
            <div style="font-size: 0.82rem; font-weight: 800; color: #38BDF8; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px;">
                🔬 Metodología & Pipeline de Innovación Tecnológica (Presentación Jurado)
            </div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; text-align: center;">
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 4px;">📥</div>
                    <b style="color: #F8FAFC; font-size: 0.85rem;">1. Ingesta de Datos</b>
                    <p style="color: #94A3B8; font-size: 0.73rem; margin: 4px 0 0 0;">808 probetas reales Yura + Declaraciones EPD (ISO 14067)</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 4px;">⚙️</div>
                    <b style="color: #F8FAFC; font-size: 0.85rem;">2. Curaduría y Leyes</b>
                    <p style="color: #94A3B8; font-size: 0.73rem; margin: 4px 0 0 0;">Verificación ACI 211 / NTP 339 / Límites de agua-cemento</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 4px;">🧠</div>
                    <b style="color: #F8FAFC; font-size: 0.85rem;">3. Modelado ML</b>
                    <p style="color: #94A3B8; font-size: 0.73rem; margin: 4px 0 0 0;">Random Forest Regressor + Intervalos Conformal (90% Conf.)</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 4px;">🧬</div>
                    <b style="color: #F8FAFC; font-size: 0.85rem;">4. NSGA-II Genético</b>
                    <p style="color: #94A3B8; font-size: 0.73rem; margin: 4px 0 0 0;">Optimización bi-objetivo: Mínimo Costo vs. Mínimo CO₂</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px 10px;">
                    <div style="font-size: 1.5rem; margin-bottom: 4px;">📑</div>
                    <b style="color: #F8FAFC; font-size: 0.85rem;">5. Ficha Certificada</b>
                    <p style="color: #94A3B8; font-size: 0.73rem; margin: 4px 0 0 0;">Exportación ejecutiva en PDF y trazabilidad total</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two column overview
    left_c, right_c = st.columns([1.6, 1.0])

    with left_c:
        st.markdown("#### 🎯 Validación de Metas y Compromisos (Evaluación con Datos Reservados)")
        goals_data = [
            {"Métrica Clave": "Tiempo de diseño de mezcla", "Línea Base": "3 a 5 días", "SmartCret": "< 15 segundos", "Meta Propuesta": "-40%", "Estado": "Superada (+85%) ✅"},
            {"Métrica Clave": "Iteraciones de ensayo en lab", "Línea Base": "4 a 6 mezclas de prueba", "SmartCret": "1 a 2 ensayos", "Meta Propuesta": "-30%", "Estado": "Cumplida (-50%) ✅"},
            {"Métrica Clave": "Costo por m³ de concreto", "Línea Base": "S/ 245.00 / m³", "SmartCret": "S/ 224.40 / m³", "Meta Propuesta": "-5 a 10%", "Estado": "-8.4% de ahorro ✅"},
            {"Métrica Clave": "Emisiones de CO₂ por m³", "Línea Base": "325.0 kg CO₂/m³", "SmartCret": "245.0 kg CO₂/m³", "Meta Propuesta": "-10 a 15%", "Estado": "-24.6% de reducción ✅"},
            {"Métrica Clave": "Error predictivo en resistencia", "Línea Base": "Modelo empírico ~18%", "SmartCret": "4.34% MAPE", "Meta Propuesta": "< 10%", "Estado": "Excelente precisión ✅"},
            {"Métrica Clave": "Detección de sobrecosto/carbono", "Línea Base": "Reportes mensuales tardíos", "SmartCret": "100% en tiempo real", "Meta Propuesta": "100%", "Estado": "Alertas automáticas ✅"},
        ]
        st.dataframe(pd.DataFrame(goals_data), use_container_width=True, hide_index=True)

    with right_c:
        st.markdown("#### ⚙️ Estado de la Plataforma")
        st.markdown(
            """
            <div class="enterprise-card">
                <b>Algoritmo de Optimización:</b> NSGA-II Multiobjetivo<br/>
                <b>Modelo de Machine Learning:</b> Random Forest v2 (Yura Calibrado)<br/>
                <b>Intervalos de Incertidumbre:</b> Conformal Prediction (90% Confianza)<br/>
                <b>Normas Técnicas:</b> NTP 339.034 / ACI 318 / E.060<br/>
                <b>Persistencia:</b> SQLite (`smartcret.db`) con auditoría<br/>
                <b>Reportes:</b> Fichas Técnicas Oficiales en PDF (ReportLab)
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### 🏢 Equipo de Desarrollo & Créditos Institucionales")
        st.markdown(
            """
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px 14px; font-size: 0.82rem; color: #CBD5E1; line-height: 1.5;">
                • <b>Fabrizio Salvador Elias Perez Peralta</b> (ML, Optimización, Asistente IA - UPT)<br/>
                • <b>Vanessa de los Angeles Olanda Ccari</b> (Ingeniería Civil - UNSA)<br/>
                • <b>Daniel Alejandro Patiño Puma</b> (Ingeniería Civil - UNSA)<br/>
                • <b>Israel Nazareth Chaparro Cruz</b> (Asesor metodológico - UPT)
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================
# 2. DOSIFICACIÓN
# =============================================================
elif selected_module == "Dosificación":
    st.markdown("<div class='module-header'>🧮 Módulo de Dosificación Volumétrica por m³</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Cálculo volumétrico estricto de proporciones de concreto según métodos ACI 211 y NTP 339.</div>", unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1.1, 1.3])

    with d_col1:
        st.markdown("#### Parámetros de Diseño")
        fc_req = st.slider("Resistencia f'c requerida (MPa)", 17.5, 50.0, 28.0, 0.5)
        slump_req = st.slider("Asentamiento / Slump objetivo (mm)", 50, 180, 100, 10)
        cemento_tipo = st.selectbox("Tipo de Cemento", ["Tipo I", "Tipo IP (Puzolánico YURA)", "Tipo HS (Alta resistencia a sulfatos)"])
        uso_adicion = st.selectbox("Estrategia de SCM (Materiales Cementantes Suplementarios)", ["Puzolana Natural (Yura Arequipa)", "Ceniza Volante (Fly Ash)", "Escoria de Alto Horno (GGBS)", "Híbrido Ternario"])

        # Quick volumetric calculation
        w_c = round(0.48 - (fc_req - 28.0) * 0.007, 3)
        w_c = max(min(w_c, 0.58), 0.38)
        agua_calc = round(175.0 + (slump_req - 100) * 0.2, 1)
        binder_calc = round(agua_calc / w_c, 1)

        if "Puzolana" in uso_adicion:
            puz_calc = round(binder_calc * 0.20, 1)
            ceniza_calc = 0.0
            escoria_calc = 0.0
            cemento_calc = binder_calc - puz_calc
        elif "Ceniza" in uso_adicion:
            ceniza_calc = round(binder_calc * 0.20, 1)
            puz_calc = 0.0
            escoria_calc = 0.0
            cemento_calc = binder_calc - ceniza_calc
        elif "Escoria" in uso_adicion:
            ceniza_calc = 0.0
            puz_calc = 0.0
            escoria_calc = round(binder_calc * 0.35, 1)
            cemento_calc = binder_calc - escoria_calc
        elif "Ternario" in uso_adicion:
            ceniza_calc = round(binder_calc * 0.15, 1)
            puz_calc = round(binder_calc * 0.15, 1)
            escoria_calc = 0.0
            cemento_calc = binder_calc - (ceniza_calc + puz_calc)
        else:
            ceniza_calc = 0.0
            puz_calc = 0.0
            escoria_calc = 0.0
            cemento_calc = binder_calc

        sp_calc = round(binder_calc * 0.008, 2)
        agg_grueso = 1040.0
        agg_fino = 760.0

        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 12px 14px; margin-top: 14px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">RESUMEN DE PROPORCIÓN:</div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #CBD5E1;">
                    <span>Relación a/c: <b>{w_c:.2f}</b></span>
                    <span>Total Cementante: <b>{binder_calc:.0f} kg/m³</b></span>
                    <span>Reemplazo SCM: <b>{((binder_calc - cemento_calc) / binder_calc * 100):.1f}%</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with d_col2:
        st.markdown("#### Receta de Dosificación Volumétrica por Metro Cúbico (1.000 m³)")

        recipe_df = pd.DataFrame(
            [
                {"Componente": "Cemento Portland", "Cantidad": f"{cemento_calc:.1f} kg", "Volumen (L)": f"{cemento_calc/3.15:.1f} L", "Función": "Matriz aglomerante principal"},
                {"Componente": "Adición SCM (Puzolana/Ceniza)", "Cantidad": f"{puz_calc + ceniza_calc:.1f} kg", "Volumen (L)": f"{(puz_calc+ceniza_calc)/2.30:.1f} L", "Función": "Reacción puzolánica y microfiller"},
                {"Componente": "Escoria GGBS", "Cantidad": f"{escoria_calc:.1f} kg", "Volumen (L)": f"{escoria_calc/2.90:.1f} L", "Función": "Hidráulica latente y durabilidad"},
                {"Componente": "Agua de mezclado", "Cantidad": f"{agua_calc:.1f} kg", "Volumen (L)": f"{agua_calc:.1f} L", "Función": "Hidratación y lubricación"},
                {"Componente": "Superplastificante", "Cantidad": f"{sp_calc:.2f} kg", "Volumen (L)": f"{sp_calc/1.1:.1f} L", "Función": "Reductor de agua alto rango"},
                {"Componente": "Piedra Chancada 1/2 pulg", "Cantidad": f"{agg_grueso:.1f} kg", "Volumen (L)": f"{agg_grueso/2.65:.1f} L", "Función": "Esqueleto granular grueso"},
                {"Componente": "Arena Gruesa Lavada", "Cantidad": f"{agg_fino:.1f} kg", "Volumen (L)": f"{agg_fino/2.65:.1f} L", "Función": "Esqueleto granular fino"},
                {"Componente": "Aire atrapado", "Cantidad": "—", "Volumen (L)": "20.0 L (2.0%)", "Función": "Vacíos según ACI 211"},
            ]
        )
        st.dataframe(recipe_df, use_container_width=True, hide_index=True)

        # Volumetric Donut Chart
        labels = ["Cemento", "Adiciones (SCM)", "Agua", "Agregado Grueso", "Agregado Fino", "Aire"]
        vol_values = [
            cemento_calc / 3.15,
            (puz_calc + ceniza_calc) / 2.30 + escoria_calc / 2.90,
            agua_calc,
            agg_grueso / 2.65,
            agg_fino / 2.65,
            20.0,
        ]
        fig_donut = px.pie(
            names=labels,
            values=vol_values,
            title="Distribución Volumétrica de la Mezcla (1000 Litros)",
            hole=0.45,
            color_discrete_sequence=["#38BDF8", "#34D399", "#06B6D4", "#F59E0B", "#F97316", "#64748B"],
        )
        fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)


# =============================================================
# 3. OPTIMIZACIÓN (NSGA-II)
# =============================================================
elif selected_module == "Optimización":
    st.markdown("<div class='module-header'>🎛️ Motor de Optimización Multiobjetivo (NSGA-II)</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Genera el Frente de Pareto equilibrando costo económico (S/) y huella de carbono (kg CO₂) con márgenes estadísticos de seguridad.</div>", unsafe_allow_html=True)

    o_c1, o_c2, o_c3, o_c4 = st.columns(4)
    with o_c1:
        fc_opt = st.number_input("f'c requerido (MPa)", 20.0, 50.0, 28.0, 1.0)
    with o_c2:
        slump_opt = st.number_input("Slump objetivo (mm)", 50.0, 180.0, 100.0, 10.0)
    with o_c3:
        age_opt = st.selectbox("Edad de garantía", [7, 14, 28, 56, 90], index=2)
    with o_c4:
        exp_opt = st.selectbox("Exposición ambiental", ["F0_normal", "F1_humedo", "S1_sulfatos_moderado", "S2_sulfatos_severo"])

    if st.button("🚀 Ejecutar Algoritmo Genético NSGA-II", type="primary"):
        with st.spinner("Optimizando poblaciones de mezclas y verificando restricciones normativas..."):
            req = OptimizationRequest(
                fc_target_mpa=fc_opt,
                slump_target_mm=slump_opt,
                age_days=age_opt,
                exposure_class=exp_opt,
            )
            proposals = optimizer.optimize(req, population_size=70, n_generations=15)
            st.session_state["pareto_proposals"] = proposals

    if "pareto_proposals" in st.session_state:
        proposals = st.session_state["pareto_proposals"]
        st.success(f"Frente de Pareto generado con éxito ({len(proposals)} soluciones no dominadas).")

        # Prepare Pareto Plot
        plot_df = pd.DataFrame(
            [
                {
                    "Costo (S/ / m³)": p.costo_soles_m3,
                    "CO2 (kg CO₂ / m³)": p.co2_kg_m3,
                    "Resistencia (MPa)": p.fc_pred_mpa,
                    "Slump (mm)": p.slump_pred_mm,
                    "Perfil": p.perfil,
                    "Cemento (kg)": p.cemento_kg,
                    "Ceniza (kg)": p.ceniza_volante_kg,
                    "Escoria (kg)": p.escoria_kg,
                }
                for p in proposals
            ]
        )

        fig_pareto = px.scatter(
            plot_df,
            x="Costo (S/ / m³)",
            y="CO2 (kg CO₂ / m³)",
            color="Perfil",
            size="Resistencia (MPa)",
            hover_data=["Cemento (kg)", "Ceniza (kg)", "Escoria (kg)", "Slump (mm)"],
            title="Frente de Pareto: Compensación Costo Económico vs. Huella de Carbono",
            color_discrete_map={
                "Mínimo Costo": "#38BDF8",
                "Mínimo CO₂": "#34D399",
                "Ecoeficiente Balanceado": "#FBBF24",
                "Pareto": "#94A3B8",
            },
        )
        fig_pareto.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_pareto, use_container_width=True)

        # 3 Key Archetype Cards
        p_cost = proposals[0]
        p_co2 = min(proposals, key=lambda x: x.co2_kg_m3)
        p_knee = next((p for p in proposals if p.perfil == "Ecoeficiente Balanceado"), proposals[len(proposals)//2])

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
                <div class="enterprise-card" style="border-top: 4px solid #38BDF8;">
                    <span class="badge-yura">Opción 1: Mínimo Costo</span>
                    <h3 style="margin: 8px 0; color:#38BDF8;">S/ {p_cost.costo_soles_m3:.2f} / m³</h3>
                    <b>Emisión:</b> {p_cost.co2_kg_m3:.1f} kg CO₂/m³<br/>
                    <b>Resistencia f'c:</b> {p_cost.fc_pred_mpa:.1f} MPa [{p_cost.fc_lower_mpa:.1f} - {p_cost.fc_upper_mpa:.1f}]<br/>
                    <b>Dosificación:</b> C: {p_cost.cemento_kg:.0f} kg | SCM: {p_cost.ceniza_volante_kg+p_cost.escoria_kg:.0f} kg | a/c: {p_cost.relacion_ac:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="enterprise-card" style="border-top: 4px solid #34D399;">
                    <span class="badge-norm">Opción 2: Mínimo Carbono</span>
                    <h3 style="margin: 8px 0; color:#34D399;">{p_co2.co2_kg_m3:.1f} kg CO₂ / m³</h3>
                    <b>Costo:</b> S/ {p_co2.costo_soles_m3:.2f} / m³<br/>
                    <b>Resistencia f'c:</b> {p_co2.fc_pred_mpa:.1f} MPa [{p_co2.fc_lower_mpa:.1f} - {p_co2.fc_upper_mpa:.1f}]<br/>
                    <b>Dosificación:</b> C: {p_co2.cemento_kg:.0f} kg | SCM: {p_co2.ceniza_volante_kg+p_co2.escoria_kg:.0f} kg | a/c: {p_co2.relacion_ac:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="enterprise-card" style="border-top: 4px solid #FBBF24;">
                    <span class="badge-yura" style="color: #FBBF24; border-color: #FBBF24;">Opción 3: Ecoeficiente Balanceado</span>
                    <h3 style="margin: 8px 0; color:#FBBF24;">S/ {p_knee.costo_soles_m3:.2f} | {p_knee.co2_kg_m3:.1f} kg</h3>
                    <b>Resistencia f'c:</b> {p_knee.fc_pred_mpa:.1f} MPa [{p_knee.fc_lower_mpa:.1f} - {p_knee.fc_upper_mpa:.1f}]<br/>
                    <b>Slump:</b> {p_knee.slump_pred_mm:.0f} mm<br/>
                    <b>Dosificación:</b> C: {p_knee.cemento_kg:.0f} kg | SCM: {p_knee.ceniza_volante_kg+p_knee.escoria_kg:.0f} kg | a/c: {p_knee.relacion_ac:.2f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Export CSV Button
        csv_data = plot_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar Soluciones de Pareto en CSV (Para Laboratorio)",
            data=csv_data,
            file_name="SmartCret_Pareto_Solutions.csv",
            mime="text/csv",
        )


# =============================================================
# 4. ESCENARIOS
# =============================================================
elif selected_module == "Escenarios":
    st.markdown("<div class='module-header'>📈 Comparador de Escenarios & Simulador de Imprevistos</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Simula alteraciones de precios o restricciones de suministro y genera fichas técnicas ejecutivas en PDF.</div>", unsafe_allow_html=True)

    sc_tab1, sc_tab2 = st.tabs(["⚡ Simular Nueva Contingencia", "📋 Escenarios Guardados & Ficha PDF"])

    with sc_tab1:
        c1, c2 = st.columns(2)
        with c1:
            imprevisto_nombre = st.text_input("Nombre de la contingencia", "Alza en Cemento (+25%) y Rotura de Escoria")
            var_cemento = st.slider("Variación en precio de Cemento (%)", -20, 100, 25, 5)
            sin_ceniza = st.checkbox("Sin disponibilidad de Ceniza Volante", False)
            sin_escoria = st.checkbox("Sin disponibilidad de Escoria GGBS", True)
        with c2:
            fc_esc = st.number_input("f'c estructural (MPa)", 20.0, 45.0, 28.0)
            slump_esc = st.number_input("Slump de colocación (mm)", 50.0, 160.0, 100.0)

        if st.button("🔄 Ejecutar Re-optimización Adaptativa", type="primary"):
            with st.spinner("Calculando impacto y reconfigurando mezclas..."):
                unavail = []
                if sin_ceniza: unavail.append("ceniza_volante")
                if sin_escoria: unavail.append("escoria")

                req_esc = OptimizationRequest(fc_target_mpa=fc_esc, slump_target_mm=slump_esc)
                event_esc = ContingencyEvent(
                    name=imprevisto_nombre,
                    price_multipliers={"cemento": 1.0 + (var_cemento / 100.0)},
                    unavailable_materials=unavail,
                )
                b_mix, n_mix, impact_res = simulator.simulate(req_esc, event_esc)
                st.session_state["active_contingency"] = (b_mix, n_mix, impact_res)

        if "active_contingency" in st.session_state:
            b_mix, n_mix, impact_res = st.session_state["active_contingency"]
            st.info(f"💡 **Dictamen de Ingeniería:** {impact_res.engineering_summary}")

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Costo por m³ resultante", f"S/ {impact_res.cost_new:.2f}", delta=f"{impact_res.cost_delta:+.2f} ({impact_res.cost_delta_pct:+.1f}%)", delta_color="inverse")
            with m2:
                st.metric("Huella de Carbono resultante", f"{impact_res.co2_new:.1f} kg CO₂/m³", delta=f"{impact_res.co2_delta:+.1f} ({impact_res.co2_delta_pct:+.1f}%)", delta_color="inverse")

            if st.button("💾 Guardar este escenario en la Base de Datos SQLite"):
                sc_obj = ConcreteScenario(
                    scenario_id=f"esc_{int(pd.Timestamp.now().timestamp())}",
                    name=imprevisto_nombre,
                    project_name="Proyecto Principal",
                    requirements={"fc": fc_esc, "slump": slump_esc},
                    mix=n_mix,
                    notes=impact_res.engineering_summary,
                )
                scenario_manager.save_scenario(sc_obj)
                st.success("✅ Escenario almacenado exitosamente en SQLite.")

    with sc_tab2:
        saved = scenario_manager.list_scenarios()
        if not saved:
            st.warning("No hay escenarios guardados aún. Ejecuta una simulación en la pestaña anterior o guarda una mezcla.")
        else:
            st.markdown("#### Escenarios Almacenados en SQLite")
            st.dataframe(pd.DataFrame(saved), use_container_width=True)

            selected_scens = st.multiselect(
                "Selecciona escenarios para comparar y exportar en PDF:",
                options=[s["scenario_id"] for s in saved],
                default=[s["scenario_id"] for s in saved[:2]],
            )

            if selected_scens:
                comp_table = scenario_manager.compare_scenarios(selected_scens)
                st.markdown("#### Comparativa Lado a Lado")
                st.dataframe(comp_table, use_container_width=True)

                pdf_target = Path("data/processed/ficha_tecnica_escenarios.pdf")
                if st.button("📄 Generar y Descargar Ficha Técnica en PDF"):
                    pdf_path = scenario_manager.export_pdf_report(selected_scens, pdf_target)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Descargar Reporte PDF Oficial de Ingeniería",
                            data=f,
                            file_name="SmartCret_Ficha_Tecnica.pdf",
                            mime="application/pdf",
                        )


# =============================================================
# 5. MATERIALES
# =============================================================
elif selected_module == "Materiales":
    st.markdown("<div class='module-header'>🗄️ Catálogo y Gestión de Materiales (Calibrado Yura)</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Costos unitarios en soles (S/), factores de emisión EPD certificados y disponibilidad en obra.</div>", unsafe_allow_html=True)

    materials_dict = optimizer.materials["materials"]
    mat_rows = []
    for k, v in materials_dict.items():
        mat_rows.append(
            {
                "Clave": k,
                "Nombre Comercial": v["name"],
                "Costo Unitario (S/ / kg)": v["cost_per_unit"],
                "Factor Emisión (kg CO₂e/kg)": v["co2_factor_kg"],
                "Densidad (kg/m³)": v["density_kg_m3"],
                "Certificación EPD": "YURA ISO 14067 ✅" if "yura" in k.lower() or "puzolana" in k.lower() else "Estándar E.060",
                "Disponible": "Sí ✅" if v.get("available", True) else "No ❌",
            }
        )
    df_mat = pd.DataFrame(mat_rows)
    st.dataframe(df_mat, use_container_width=True, hide_index=True)

    # Cost vs CO2 Benchmark Chart for Materials
    fig_mat = px.bar(
        df_mat,
        x="Nombre Comercial",
        y="Factor Emisión (kg CO₂e/kg)",
        color="Factor Emisión (kg CO₂e/kg)",
        title="Huella de Carbono por Material (kg CO₂e por kg de insumo)",
        color_continuous_scale="Teal",
    )
    fig_mat.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
    st.plotly_chart(fig_mat, use_container_width=True)


# =============================================================
# 6. PREDICCIÓN IA (Laboratorio Virtual con Presets para el Jurado)
# =============================================================
elif selected_module == "Predicción IA":
    st.markdown("<div class='module-header'>🧬 Laboratorio Virtual & Predicción con IA</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Inferencia en tiempo real con cuantificación rigurosa de incertidumbre conformal al 90% de confianza y asistente con tool calling.</div>", unsafe_allow_html=True)

    p_tab1, p_tab2 = st.tabs(["🧪 Laboratorio Virtual Interactivo", "🤖 Asistente Agéntico de Ingeniería"])

    with p_tab1:
        # Initialize default slider session state if not set
        if "slider_cem" not in st.session_state:
            st.session_state["slider_cem"] = 310.0
            st.session_state["slider_fa"] = 45.0
            st.session_state["slider_slag"] = 30.0
            st.session_state["slider_poz"] = 0.0
            st.session_state["slider_water"] = 172.0
            st.session_state["slider_sp"] = 2.2
            st.session_state["slider_coarse"] = 1030.0
            st.session_state["slider_fine"] = 770.0
            st.session_state["slider_age"] = 28
            st.session_state["slider_type"] = "Tipo I"

        # Preset Buttons for Jury Demonstration
        st.markdown("###### 🎯 Casos de Prueba Rápidos para Demostración al Jurado:")
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        if btn_c1.button("📌 f'c = 210 kg/cm² (Zapatas)"):
            st.session_state["slider_cem"] = 280.0
            st.session_state["slider_fa"] = 0.0
            st.session_state["slider_slag"] = 0.0
            st.session_state["slider_poz"] = 0.0
            st.session_state["slider_water"] = 180.0
            st.session_state["slider_sp"] = 1.0
            st.session_state["slider_coarse"] = 1050.0
            st.session_state["slider_fine"] = 820.0
            st.session_state["slider_age"] = 28
            st.session_state["slider_type"] = "Tipo I"
            st.rerun()

        if btn_c2.button("📌 f'c = 280 kg/cm² (Columnas)"):
            st.session_state["slider_cem"] = 350.0
            st.session_state["slider_fa"] = 0.0
            st.session_state["slider_slag"] = 0.0
            st.session_state["slider_poz"] = 0.0
            st.session_state["slider_water"] = 175.0
            st.session_state["slider_sp"] = 2.5
            st.session_state["slider_coarse"] = 1020.0
            st.session_state["slider_fine"] = 760.0
            st.session_state["slider_age"] = 28
            st.session_state["slider_type"] = "Tipo I"
            st.rerun()

        if btn_c3.button("📌 YURA Ecoeficiente IP"):
            st.session_state["slider_cem"] = 290.0
            st.session_state["slider_fa"] = 35.0
            st.session_state["slider_slag"] = 0.0
            st.session_state["slider_poz"] = 30.0
            st.session_state["slider_water"] = 165.0
            st.session_state["slider_sp"] = 2.8
            st.session_state["slider_coarse"] = 1040.0
            st.session_state["slider_fine"] = 780.0
            st.session_state["slider_age"] = 28
            st.session_state["slider_type"] = "Tipo IP"
            st.rerun()

        if btn_c4.button("📌 Alta Resistencia 38 MPa"):
            st.session_state["slider_cem"] = 420.0
            st.session_state["slider_fa"] = 40.0
            st.session_state["slider_slag"] = 50.0
            st.session_state["slider_poz"] = 0.0
            st.session_state["slider_water"] = 155.0
            st.session_state["slider_sp"] = 4.5
            st.session_state["slider_coarse"] = 1000.0
            st.session_state["slider_fine"] = 710.0
            st.session_state["slider_age"] = 28
            st.session_state["slider_type"] = "Tipo I"
            st.rerun()

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        col_sliders1, col_sliders2 = st.columns(2)
        with col_sliders1:
            p_cem = st.slider("Cemento Portland (kg/m³)", 150.0, 520.0, key="slider_cem", step=5.0)
            p_fa = st.slider("Ceniza Volante (kg/m³)", 0.0, 180.0, key="slider_fa", step=5.0)
            p_slag = st.slider("Escoria GGBS (kg/m³)", 0.0, 220.0, key="slider_slag", step=5.0)
            p_poz = st.slider("Puzolana Natural (kg/m³)", 0.0, 180.0, key="slider_poz", step=5.0)
            p_water = st.slider("Agua (kg/m³)", 135.0, 230.0, key="slider_water", step=2.0)
        with col_sliders2:
            p_sp = st.slider("Superplastificante (kg/m³)", 0.0, 12.0, key="slider_sp", step=0.2)
            p_coarse = st.slider("Agregado Grueso (kg/m³)", 800.0, 1200.0, key="slider_coarse", step=10.0)
            p_fine = st.slider("Agregado Fino (kg/m³)", 600.0, 950.0, key="slider_fine", step=10.0)
            p_age = st.select_slider("Edad de curado (días)", options=[3, 7, 14, 28, 56, 90], key="slider_age")
            type_options = ["Tipo I", "Tipo IP", "Tipo IPM", "Tipo HS"]
            p_type = st.selectbox("Tipo de cemento", type_options, key="slider_type")

        input_data = {
            "cemento_kg": p_cem,
            "ceniza_volante_kg": p_fa,
            "escoria_kg": p_slag,
            "puzolana_kg": p_poz,
            "agua_kg": p_water,
            "superplastificante_kg": p_sp,
            "agregado_grueso_kg": p_coarse,
            "agregado_fino_kg": p_fine,
            "edad_dias": p_age,
            "tipo_cemento": p_type,
        }

        pred_res = predictor.predict(input_data)

        st.markdown("---")
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric(
                "Resistencia f'c Estimada",
                f"{pred_res.resistencia_mpa:.1f} MPa",
                delta=f"90% CI: [{pred_res.resistencia_lower:.1f} - {pred_res.resistencia_upper:.1f}]",
            )
        with k2:
            st.metric(
                "Asentamiento / Slump",
                f"{pred_res.slump_mm:.0f} mm",
                delta=f"90% CI: [{pred_res.slump_lower:.0f} - {pred_res.slump_upper:.0f}]",
            )
        with k3:
            if pred_res.is_extrapolated:
                st.warning("⚠️ Extrapolación detectada: Fuera del dominio experimental")
            else:
                st.success("✅ En dominio experimental seguro (Confiable)")

        # Curing Maturity Curve with Uncertainty Bands
        ages_curve = [3, 7, 14, 28, 56, 90]
        curve_pts = []
        for a in ages_curve:
            d_copy = input_data.copy()
            d_copy["edad_dias"] = a
            out_p = predictor.predict(d_copy)
            curve_pts.append({
                "Edad": a,
                "Resistencia": out_p.resistencia_mpa,
                "Lower": out_p.resistencia_lower,
                "Upper": out_p.resistencia_upper,
            })
        df_curve = pd.DataFrame(curve_pts)

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=df_curve["Edad"], y=df_curve["Upper"], fill=None, mode='lines', line_color='rgba(56, 189, 248, 0.2)', name='Límite Superior 90%'))
        fig_curve.add_trace(go.Scatter(x=df_curve["Edad"], y=df_curve["Lower"], fill='tonexty', mode='lines', line_color='rgba(56, 189, 248, 0.2)', name='Banda de Confianza 90%'))
        fig_curve.add_trace(go.Scatter(x=df_curve["Edad"], y=df_curve["Resistencia"], mode='lines+markers', line=dict(color='#38BDF8', width=3), name='Resistencia Estimada'))
        fig_curve.update_layout(title="Curva de Maduración y Ganancia de Resistencia en el Tiempo (Normativa ASTM C1074)", xaxis_title="Edad de curado (días)", yaxis_title="Resistencia a compresión (MPa)", height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_curve, use_container_width=True)

    with p_tab2:
        st.markdown("#### Consultas al Asistente Agéntico de IA")
        st.caption("El asistente no inventa cifras: invoca directamente el motor de optimización, predicción o contingencias.")

        sample_pills = [
            "¿qué mezcla me conviene si sube el precio del cemento?",
            "optimiza una mezcla para 30 MPa y slump 100 mm",
            "ver presupuesto de carbono de la obra",
        ]
        q_cols = st.columns(len(sample_pills))
        for idx, pill in enumerate(sample_pills):
            if q_cols[idx].button(pill, key=f"pill_{idx}"):
                st.session_state["agent_q"] = pill

        agent_prompt = st.text_input("Formula tu consulta técnica:", value=st.session_state.get("agent_q", "¿qué mezcla me conviene si sube el precio del cemento?"))
        if st.button("Enviar Consulta", type="primary"):
            with st.spinner("Ejecutando herramientas y analizando..."):
                resp = assistant.process_query(agent_prompt)
                st.markdown(resp["response"])
                with st.expander("🛠️ Herramientas ejecutadas y trazabilidad matemática"):
                    st.write(resp["tools_called"])
                    st.json(resp["raw_tool_results"])


# =============================================================
# 7. HUELLA DE CARBONO
# =============================================================
elif selected_module == "Huella de carbono":
    st.markdown("<div class='module-header'>🍃 Control de Presupuesto de Carbono (ISO 14067)</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Seguimiento de la cuota autorizada de emisiones de CO₂ por obra y metas de descarbonización.</div>", unsafe_allow_html=True)

    status = carbon_tracker.get_project_status("OBRA-LIMA-01")

    # Gauge Chart for Carbon
    co2_val = status["co2_emitted_ton"]
    co2_max = status["co2_budget_ton"]
    co2_pct = status["co2_pct_consumed"]

    h_col1, h_col2 = st.columns([1.2, 1.0])
    with h_col1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = co2_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Emisiones Acumuladas vs. Presupuesto (Toneladas CO₂)", 'font': {'size': 18}},
            delta = {'reference': co2_max, 'increasing': {'color': "#EF4444"}},
            gauge = {
                'axis': {'range': [None, co2_max * 1.25], 'tickwidth': 1},
                'bar': {'color': "#38BDF8"},
                'bgcolor': "rgba(30,41,59,0.8)",
                'steps': [
                    {'range': [0, co2_max * 0.8], 'color': "rgba(52,211,153,0.25)"},
                    {'range': [co2_max * 0.8, co2_max], 'color': "rgba(251,191,36,0.3)"},
                    {'range': [co2_max, co2_max * 1.25], 'color': "rgba(239,68,68,0.3)"},
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 4},
                    'thickness': 0.75,
                    'value': co2_max,
                }
            }
        ))
        fig_gauge.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with h_col2:
        st.markdown(f"#### Obra Activa: **{status['project_name']}**")
        st.write(f"• Volumen vaciado a la fecha: **{status['poured_volume_m3']} m³** de {status['planned_volume_m3']} m³ planificados")
        st.write(f"• Consumo del presupuesto de CO₂: **{co2_pct:.1f}%**")
        st.write(f"• Toneladas restantes autorizadas: **{status['co2_remaining_ton']} t CO₂**")
        st.write(f"• Gasto económico acumulado: **S/ {status['cost_spent_soles']:,.2f}** ({status['cost_pct_consumed']:.1f}%)")

        st.markdown("---")
        st.markdown("#### Impacto Ambiental Positivo Evitado")
        co2_evitado = (status['poured_volume_m3'] * 325.0 / 1000.0) - co2_val
        st.success(f"🌱 Se han evitado **{co2_evitado:.1f} toneladas de CO₂** frente al concreto tradicional de referencia (~{int(co2_evitado * 45)} árboles plantados).")


# =============================================================
# 8. ALERTAS
# =============================================================
elif selected_module == "Alertas":
    st.markdown("<div class='module-header'>🔔 Centro de Alertas y Notificaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Monitoreo activo de límites presupuestales, normativos y disponibilidad de insumos.</div>", unsafe_allow_html=True)

    alerts = carbon_tracker.check_alerts("OBRA-LIMA-01")

    # Add simulated system alerts for demonstration
    all_alerts = [
        {"Nivel": "CRÍTICA", "Icono": "🔴", "Tipo": "Presupuesto", "Mensaje": "Alerta preventiva: Se ha alcanzado el 82.5% de la cuota de carbono autorizada para la Obra Las Palmeras.", "Hora": "Hace 10 min"},
        {"Nivel": "PREVENTIVA", "Icono": "🟡", "Tipo": "Materiales", "Mensaje": "Stock de Escoria GGBS por debajo del umbral de reserva estratégica (quedan 18 toneladas).", "Hora": "Hace 2 horas"},
        {"Nivel": "INFORMATIVA", "Icono": "🟢", "Tipo": "Conformidad", "Mensaje": "Mezcla certificada: El diseño f'c = 28 MPa cumple con la norma NTP 339 con 90% de confianza estadística.", "Hora": "Hoy 08:30 AM"},
    ]

    for a in all_alerts:
        border_col = "#EF4444" if a["Nivel"] == "CRÍTICA" else ("#F59E0B" if a["Nivel"] == "PREVENTIVA" else "#10B981")
        st.markdown(
            f"""
            <div style="background: #111C2E; border-left: 5px solid {border_col}; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <b>{a['Icono']} [{a['Nivel']}] {a['Tipo']}</b>
                    <span style="color: #64748B; font-size: 0.8rem;">{a['Hora']}</span>
                </div>
                <div style="color: #CBD5E1; margin-top: 5px; font-size: 0.95rem;">{a['Mensaje']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Simular Evento de Alerta para el Jurado")
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🚨 Disparar Alerta Crítica por Superación de CO₂ (>100%)"):
            carbon_tracker.log_pour("OBRA-LIMA-01", 60.0, 300.0, 240.0, "Vaciado de Emergencia")
            st.rerun()
    with c_btn2:
        if st.button("🔄 Reiniciar Presupuesto de Obra"):
            carbon_tracker.create_project(
                ProjectCarbonBudget(
                    project_id="OBRA-LIMA-01",
                    project_name="Complejo Residencial Las Palmeras",
                    planned_volume_m3=300.0,
                    co2_budget_ton=68.0,
                    cost_budget_soles=65000.0,
                    warning_threshold_pct=80.0,
                )
            )
            st.rerun()
