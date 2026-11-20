"""Streamlit Web Application for SmartCret - Enterprise Concrete Optimization Platform.

Enterprise visual interface engineered for institutional and jury evaluation:
Visual-first architecture replacing dense text with intuitive Plotly graphs,
radar charts, gauges, and comparison plots under the ACI 318 / ACI 211.1 framework.
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
from plotly.subplots import make_subplots
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
        padding: 20px 26px;
        margin-bottom: 20px;
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
    
    /* Institutional Badges */
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
    .kpi-card {
        background: #111C2E;
        border: 1px solid #1E2E48;
        border-radius: 12px;
        padding: 18px 20px;
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
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 6px;
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
        font-size: 1.55rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 4px;
        letter-spacing: -0.02em;
    }
    .module-sub {
        color: #94A3B8;
        font-size: 0.92rem;
        margin-bottom: 18px;
        font-weight: 400;
    }

    /* Enterprise Visual Card */
    .chart-card {
        background: #111C2E;
        border: 1px solid #1E2E48;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.2);
    }
    .chart-card-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
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
        "Asistente IA",
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
        "Asistente IA": "🤖",
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
                    <span>Código Activo:</span>
                    <b style="color: #F8FAFC;">ACI 318-19 / ACI 211.1</b>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Dataset Base:</span>
                    <b style="color: #38BDF8;">Yura Arequipa (808 cil. ASTM C39)</b>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Precisión Test:</span>
                    <b style="color: #34D399;">MAPE = 4.34% (&lt; 10% ACI 214R)</b>
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
            <div class="hero-subtitle">Sistema Integral de Diseño, Optimización y Control Ecoeficiente de Concreto según Código ACI 318 y ACI 211.1</div>
        </div>
        <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 8px;">
            <div style="display: flex; gap: 8px;">
                <span class="badge-yura">🏢 Cementos YURA S.A. (ASTM C595)</span>
                <span class="badge-norm">⚖️ CÓDIGO ACI 318-19 / ACI 211.1</span>
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

# -------------------------------------------------------------
# INTERACTIVE AGENTIC ASSISTANT COMPONENT
# -------------------------------------------------------------
def render_interactive_assistant(key_prefix=""):
    st.markdown("<div class='module-header'>🤖 Asistente Agéntico de Ingeniería & Normativa ACI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='module-sub'>Interacción conversacional con ejecución verificada de herramientas matemáticas: "
        "Optimización NSGA-II, Predicción Conformal ML y Simulación de Contingencias. Conectado a la base de datos de <b>Cementos YURA</b>.</div>",
        unsafe_allow_html=True
    )

    # Initialize chat history in session state
    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 **¡Hola! Soy el Asistente Agéntico de Ingeniería de SmartCret.**\n\n"
                    "Estoy conectado a los motores de **Optimización Multiobjetivo (NSGA-II)**, "
                    "**Predicción Conformal ML (ASTM C39 / C143)** y **Simulación de Contingencias** bajo normativa "
                    "**ACI 318-19 / ACI 211.1** con la base de datos certificada de **Cementos YURA S.A.**\n\n"
                    "**¿En qué puedo ayudarte hoy?** Puedes escribir una consulta técnica en el campo inferior o tocar una de las sugerencias preconfiguradas para ver la ejecución algorítmica en tiempo real."
                ),
                "tools": [],
                "raw": [],
                "suggestions": [
                    "¿Qué mezcla me conviene si sube el precio del cemento?",
                    "Recomienda una dosificación para zapatas (210 kg/cm²)",
                    "Optimiza una mezcla para columnas f'c = 28 MPa",
                    "Ver presupuesto de carbono de la obra",
                    "¿Por qué es superior el Cemento Yura Tipo IP frente a sulfatos?",
                ]
            }
        ]

    # Executive Status Banner & Controls
    c_ctrl1, c_ctrl2 = st.columns([3, 1])
    with c_ctrl1:
        st.markdown(
            '<span class="badge-active">🟢 MOTOR AGÉNTICO ACTIVO</span> '
            '<span class="badge-norm">ACI 318-19 / ASTM C39 / ISO 14067</span> '
            '<span style="font-size:0.75rem; color:#94A3B8; margin-left:6px;">Calibrado con probetas de Arequipa</span>',
            unsafe_allow_html=True
        )
    with c_ctrl2:
        if st.button("🧹 Reiniciar Conversación", key=f"{key_prefix}_clear_chat", use_container_width=True):
            st.session_state.pop("agent_messages", None)
            st.session_state.pop("pending_query", None)
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Quick Case Chips for Immediate Demonstration to the Jury
    st.markdown("###### 🎯 Casos Rápidos de Prueba para el Jurado:")
    chip_cols = st.columns(4)
    if chip_cols[0].button("📈 ¿Qué pasa si sube el cemento?", key=f"{key_prefix}_chip_1", use_container_width=True):
        st.session_state["pending_query"] = "¿qué mezcla me conviene si sube el precio del cemento?"
        st.rerun()
    if chip_cols[1].button("🏛️ Mezcla para zapatas (210)", key=f"{key_prefix}_chip_2", use_container_width=True):
        st.session_state["pending_query"] = "recomienda una mezcla para zapatas"
        st.rerun()
    if chip_cols[2].button("🏢 Mezcla para columnas (280)", key=f"{key_prefix}_chip_3", use_container_width=True):
        st.session_state["pending_query"] = "optimiza una mezcla para columnas f'c = 28 MPa"
        st.rerun()
    if chip_cols[3].button("🍃 Presupuesto de Carbono", key=f"{key_prefix}_chip_4", use_container_width=True):
        st.session_state["pending_query"] = "ver presupuesto de carbono de la obra"
        st.rerun()

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Render Conversation Messages
    for m_idx, msg in enumerate(st.session_state["agent_messages"]):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👷"):
                st.markdown(f"**Ingeniero:** {msg['content']}")
        else:
            with st.chat_message("assistant", avatar="🤖"):
                if msg.get("tools"):
                    for t in msg["tools"]:
                        st.markdown(f'<span class="badge-active" style="font-size:0.68rem; margin-right:4px;">⚡ Herramienta Ejecutada: {t}</span>', unsafe_allow_html=True)
                    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

                st.markdown(msg["content"])

                if msg.get("raw") and len(msg["raw"]) > 0:
                    with st.expander("🛠️ Ver Trazabilidad Matemática & Datos Crudos de Herramientas"):
                        st.json(msg["raw"])

                if msg.get("suggestions"):
                    st.markdown("<div style='font-size: 0.72rem; color: #94A3B8; font-weight: 700; margin-top: 10px; margin-bottom: 6px; text-transform: uppercase;'>💡 Sugerencias rápidas de seguimiento:</div>", unsafe_allow_html=True)
                    sug_cols = st.columns(min(len(msg["suggestions"]), 4))
                    for s_idx, sug in enumerate(msg["suggestions"][:4]):
                        if sug_cols[s_idx].button(f"👉 {sug}", key=f"{key_prefix}_sug_{m_idx}_{s_idx}", use_container_width=True):
                            st.session_state["pending_query"] = sug
                            st.rerun()

    # Chat Input Box
    user_input = st.chat_input("Escribe una consulta de ingeniería, solicita un diseño ACI o formula un imprevisto...", key=f"{key_prefix}_chat_input")
    query_to_process = None
    if user_input:
        query_to_process = user_input
    elif "pending_query" in st.session_state and st.session_state["pending_query"]:
        query_to_process = st.session_state.pop("pending_query")

    if query_to_process:
        st.session_state["agent_messages"].append({
            "role": "user",
            "content": query_to_process,
        })
        with st.spinner("Ejecutando herramientas algorítmicas y verificando límites ACI..."):
            try:
                resp = assistant.process_query(query_to_process)
                st.session_state["agent_messages"].append({
                    "role": "assistant",
                    "content": resp["response"],
                    "tools": resp["tools_called"],
                    "raw": resp["raw_tool_results"],
                    "suggestions": resp.get("suggestions", []),
                })
            except Exception as e:
                st.session_state["agent_messages"].append({
                    "role": "assistant",
                    "content": (
                        f"⚠️ **Aviso del Asistente:** No fue posible procesar la consulta directamente ({type(e).__name__}: {str(e)}).\n\n"
                        f"Te recomiendo seleccionar una de las consultas de prueba recomendadas abajo o ingresar especificaciones concretas (por ejemplo: *optimiza mezcla para 28 MPa* o *¿qué mezcla conviene si sube el cemento?*)."
                    ),
                    "tools": [],
                    "raw": [],
                    "suggestions": [
                        "¿Qué mezcla me conviene si sube el precio del cemento?",
                        "Optimiza una mezcla para columnas f'c = 28 MPa",
                        "Ver presupuesto de carbono de la obra",
                        "Recomienda una dosificación para zapatas (210 kg/cm²)",
                    ],
                })
        st.rerun()


# 1. PANEL PRINCIPAL (Vista Visual de Alto Impacto para el Jurado)
# =============================================================
if selected_module == "Panel principal":
    st.markdown("<div class='module-header'>⊞ Panel Principal de Control de Ingeniería</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Evaluación visual de desempeño: reducción de costos, descarbonización y precisión del modelo calibrado con datos de Cementos YURA.</div>", unsafe_allow_html=True)

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

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # VISUAL COMPARISON SECTION: Benchmark Bars + Multidimensional Radar
    vis_c1, vis_c2 = st.columns([1.3, 1.0])

    with vis_c1:
        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">📊 Comparativa de Impacto Directo: Tradicional vs. SmartCret IA</div>
            """,
            unsafe_allow_html=True,
        )

        fig_bench = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Costo por m³ de Concreto (S/ / m³)", "Huella de Carbono (kg CO₂ / m³)"),
            horizontal_spacing=0.15
        )

        # Bar 1: Cost
        fig_bench.add_trace(
            go.Bar(
                name="Tradicional",
                x=["Costo"],
                y=[245.0],
                marker_color="#64748B",
                text=["S/ 245.00"],
                textposition="auto",
            ),
            row=1, col=1
        )
        fig_bench.add_trace(
            go.Bar(
                name="SmartCret IA",
                x=["Costo"],
                y=[224.40],
                marker_color="#38BDF8",
                text=["S/ 224.40 (-8.4%)"],
                textposition="auto",
            ),
            row=1, col=1
        )

        # Bar 2: CO2
        fig_bench.add_trace(
            go.Bar(
                name="Tradicional",
                x=["CO₂"],
                y=[325.0],
                marker_color="#64748B",
                text=["325 kg"],
                textposition="auto",
                showlegend=False,
            ),
            row=1, col=2
        )
        fig_bench.add_trace(
            go.Bar(
                name="SmartCret IA",
                x=["CO₂"],
                y=[245.0],
                marker_color="#34D399",
                text=["245 kg (-24.6%)"],
                textposition="auto",
                showlegend=False,
            ),
            row=1, col=2
        )

        fig_bench.update_layout(
            barmode="group",
            height=280,
            margin=dict(t=40, b=20, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bench, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with vis_c2:
        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">🕸️ Evaluación Multidimensional (Radar de Rendimiento)</div>
            """,
            unsafe_allow_html=True,
        )

        radar_categories = [
            'Velocidad Diseño',
            'Eficiencia Costo',
            'Descarbonización',
            'Precisión ML',
            'Conformidad ACI 318'
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[9.5, 8.5, 9.0, 9.6, 10.0],
            theta=radar_categories,
            fill='toself',
            name='SmartCret (Yura)',
            line_color='#38BDF8',
            fillcolor='rgba(56, 189, 248, 0.35)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[2.0, 4.5, 3.2, 5.0, 7.5],
            theta=radar_categories,
            fill='toself',
            name='Diseño Tradicional',
            line_color='#94A3B8',
            fillcolor='rgba(148, 163, 184, 0.15)'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=8, color="#94A3B8")),
                angularaxis=dict(tickfont=dict(size=9, color="#CBD5E1")),
                bgcolor='rgba(15,23,42,0.4)',
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=280,
            margin=dict(t=25, b=25, l=35, r=35),
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Visual Workflow Stepper Bar
    st.markdown(
        """
        <div class="chart-card">
            <div class="chart-card-title">🔬 Metodología & Pipeline de Innovación Tecnológica (Presentación Jurado)</div>
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; text-align: center; margin-top: 8px;">
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 8px;">
                    <span style="font-size: 1.4rem;">📥</span><br/>
                    <b style="color: #38BDF8; font-size: 0.8rem;">1. Ingesta Yura</b>
                    <p style="color: #94A3B8; font-size: 0.7rem; margin: 3px 0 0 0;">808 probetas reales + EPD ISO 14067</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 8px;">
                    <span style="font-size: 1.4rem;">⚙️</span><br/>
                    <b style="color: #38BDF8; font-size: 0.8rem;">2. ACI 211 / 318</b>
                    <p style="color: #94A3B8; font-size: 0.7rem; margin: 3px 0 0 0;">Volumetría estricta y límites w/cm</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 8px;">
                    <span style="font-size: 1.4rem;">🧠</span><br/>
                    <b style="color: #38BDF8; font-size: 0.8rem;">3. Modelado ML</b>
                    <p style="color: #94A3B8; font-size: 0.7rem; margin: 3px 0 0 0;">Random Forest + Conformal 90%</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 8px;">
                    <span style="font-size: 1.4rem;">🧬</span><br/>
                    <b style="color: #38BDF8; font-size: 0.8rem;">4. Genético NSGA-II</b>
                    <p style="color: #94A3B8; font-size: 0.7rem; margin: 3px 0 0 0;">Frente de Pareto Costo vs. CO₂</p>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px; padding: 12px 8px;">
                    <span style="font-size: 1.4rem;">📑</span><br/>
                    <b style="color: #38BDF8; font-size: 0.8rem;">5. Ficha ACI 301</b>
                    <p style="color: #94A3B8; font-size: 0.7rem; margin: 3px 0 0 0;">Exportación ejecutiva en PDF</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ Ver Créditos Académicos y Equipo de Investigación (UPT - UNSA)"):
        st.markdown(
            """
            • **Fabrizio Salvador Elias Perez Peralta** (ML, Optimización, Asistente IA - Universidad Privada de Tacna)<br/>
            • **Vanessa de los Angeles Olanda Ccari** (Ingeniería Civil - Universidad Nacional de San Agustín de Arequipa)<br/>
            • **Daniel Alejandro Patiño Puma** (Ingeniería Civil - Universidad Nacional de San Agustín de Arequipa)<br/>
            • **Israel Nazareth Chaparro Cruz** (Asesor metodológico - Universidad Privada de Tacna)
            """,
            unsafe_allow_html=True,
        )


# =============================================================
# 2. DOSIFICACIÓN (Visualización ACI 211.1 y Gauge de w/cm)
# =============================================================
elif selected_module == "Dosificación":
    st.markdown("<div class='module-header'>🧮 Módulo de Dosificación Volumétrica por m³</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Cálculo volumétrico estricto de proporciones según el estándar ACI 211.1 y verificación de requisitos de durabilidad ACI 318.</div>", unsafe_allow_html=True)

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

        # GAUGE w/cm ACI 318
        st.markdown(
            """
            <div class="chart-card" style="margin-top: 14px;">
                <div class="chart-card-title">🎯 Control de Durabilidad: Relación w/cm (ACI 318 Tabla 19.3.2.1)</div>
            """,
            unsafe_allow_html=True,
        )
        fig_wc = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = w_c,
            gauge = {
                'axis': {'range': [0.30, 0.65], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
                'bar': {'color': '#38BDF8'},
                'bgcolor': 'rgba(15,23,42,0.8)',
                'steps': [
                    {'range': [0.30, 0.45], 'color': 'rgba(52,211,153,0.35)'},
                    {'range': [0.45, 0.50], 'color': 'rgba(56,189,248,0.25)'},
                    {'range': [0.50, 0.55], 'color': 'rgba(251,191,36,0.3)'},
                    {'range': [0.55, 0.65], 'color': 'rgba(239,68,68,0.35)'},
                ],
                'threshold': {
                    'line': {'color': '#EF4444', 'width': 3},
                    'thickness': 0.8,
                    'value': 0.50
                }
            }
        ))
        fig_wc.update_layout(height=170, margin=dict(t=25, b=10, l=25, r=25), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wc, use_container_width=True)
        st.caption("🟢 Verde: Cumple exposición severa (w/cm ≤ 0.45) | 🟡 Amarillo: Exposición moderada | 🔴 Rojo: Excede límite")
        st.markdown("</div>", unsafe_allow_html=True)

    with d_col2:
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
        
        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">🥧 Distribución Volumétrica Absoluta de la Mezcla (1000 Litros - ACI 211.1)</div>
            """,
            unsafe_allow_html=True,
        )
        fig_donut = px.pie(
            names=labels,
            values=vol_values,
            hole=0.45,
            color_discrete_sequence=["#38BDF8", "#34D399", "#06B6D4", "#F59E0B", "#F97316", "#64748B"],
        )
        fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=240, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recipe Table
        with st.expander("📋 Ver Tabla Detallada de Proporciones y Especificaciones ASTM", expanded=True):
            recipe_df = pd.DataFrame(
                [
                    {"Componente": "Cemento Portland", "Cantidad": f"{cemento_calc:.1f} kg", "Volumen (L)": f"{cemento_calc/3.15:.1f} L", "Especificación": "ASTM C150 / ASTM C595"},
                    {"Componente": "Adición SCM (Puzolana/Ceniza)", "Cantidad": f"{puz_calc + ceniza_calc:.1f} kg", "Volumen (L)": f"{(puz_calc+ceniza_calc)/2.30:.1f} L", "Especificación": "ASTM C618 / ACI 318"},
                    {"Componente": "Escoria GGBS", "Cantidad": f"{escoria_calc:.1f} kg", "Volumen (L)": f"{escoria_calc/2.90:.1f} L", "Especificación": "ASTM C989"},
                    {"Componente": "Agua de mezclado", "Cantidad": f"{agua_calc:.1f} kg", "Volumen (L)": f"{agua_calc:.1f} L", "Especificación": "ASTM C1602"},
                    {"Componente": "Superplastificante", "Cantidad": f"{sp_calc:.2f} kg", "Volumen (L)": f"{sp_calc/1.1:.1f} L", "Especificación": "ASTM C494 Tipo F"},
                    {"Componente": "Piedra Chancada 1/2 pulg", "Cantidad": f"{agg_grueso:.1f} kg", "Volumen (L)": f"{agg_grueso/2.65:.1f} L", "Especificación": "ASTM C33 (Grueso)"},
                    {"Componente": "Arena Gruesa Lavada", "Cantidad": f"{agg_fino:.1f} kg", "Volumen (L)": f"{agg_fino/2.65:.1f} L", "Especificación": "ASTM C33 (Fino)"},
                    {"Componente": "Aire atrapado", "Cantidad": "—", "Volumen (L)": "20.0 L (2.0%)", "Especificación": "ACI 211.1 Tabla 6.3.3"},
                ]
            )
            st.dataframe(recipe_df, use_container_width=True, hide_index=True)


# =============================================================
# 3. OPTIMIZACIÓN (NSGA-II con Gráfico Comparativo de Arquetipos)
# =============================================================
elif selected_module == "Optimización":
    st.markdown("<div class='module-header'>🎛️ Motor de Optimización Multiobjetivo (NSGA-II)</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Genera el Frente de Pareto equilibrando costo (S/) y huella de carbono (kg CO₂) garantizando f'cr según ACI 214R y durabilidad ACI 318.</div>", unsafe_allow_html=True)

    o_c1, o_c2, o_c3, o_c4 = st.columns(4)
    with o_c1:
        fc_opt = st.number_input("f'c requerido (MPa)", 20.0, 50.0, 28.0, 1.0)
    with o_c2:
        slump_opt = st.number_input("Slump objetivo (mm)", 50.0, 180.0, 100.0, 10.0)
    with o_c3:
        age_opt = st.selectbox("Edad de garantía", [7, 14, 28, 56, 90], index=2)
    with o_c4:
        exp_opt = st.selectbox("Exposición ambiental ACI 318", ["F0_normal", "F1_humedo", "S1_sulfatos_moderado", "S2_sulfatos_severo"])

    if st.button("🚀 Ejecutar Algoritmo Genético NSGA-II", type="primary"):
        with st.spinner("Optimizando poblaciones de mezclas y verificando restricciones normativas ACI..."):
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

        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">📈 Frente de Pareto: Trade-off Multiobjetivo Costo Económico vs. Emisión de CO₂</div>
            """,
            unsafe_allow_html=True,
        )
        fig_pareto = px.scatter(
            plot_df,
            x="Costo (S/ / m³)",
            y="CO2 (kg CO₂ / m³)",
            color="Perfil",
            size="Resistencia (MPa)",
            hover_data=["Cemento (kg)", "Ceniza (kg)", "Escoria (kg)", "Slump (mm)"],
            color_discrete_map={
                "Mínimo Costo": "#38BDF8",
                "Mínimo CO₂": "#34D399",
                "Ecoeficiente Balanceado": "#FBBF24",
                "Pareto": "#94A3B8",
            },
        )
        fig_pareto.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_pareto, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # 3 Key Archetype Selection
        p_cost = proposals[0]
        p_co2 = min(proposals, key=lambda x: x.co2_kg_m3)
        p_knee = next((p for p in proposals if p.perfil == "Ecoeficiente Balanceado"), proposals[len(proposals)//2])

        # VISUAL COMPARISON CHART OF THE 3 ARCHETYPES
        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">📊 Comparativa Visual de los 3 Arquetipos de Decisión Ejecutiva</div>
            """,
            unsafe_allow_html=True,
        )
        fig_arch = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Costo Unitario (S/ / m³)", "Emisión CO₂ (kg / m³)", "Resistencia f'c (MPa)"),
            horizontal_spacing=0.1
        )
        arch_names = ["Mín. Costo", "Mín. CO₂", "Ecoeficiente"]
        arch_colors = ["#38BDF8", "#34D399", "#FBBF24"]

        fig_arch.add_trace(
            go.Bar(
                x=arch_names,
                y=[p_cost.costo_soles_m3, p_co2.costo_soles_m3, p_knee.costo_soles_m3],
                marker_color=arch_colors,
                text=[f"S/ {p_cost.costo_soles_m3:.1f}", f"S/ {p_co2.costo_soles_m3:.1f}", f"S/ {p_knee.costo_soles_m3:.1f}"],
                textposition="auto",
                showlegend=False
            ),
            row=1, col=1
        )
        fig_arch.add_trace(
            go.Bar(
                x=arch_names,
                y=[p_cost.co2_kg_m3, p_co2.co2_kg_m3, p_knee.co2_kg_m3],
                marker_color=arch_colors,
                text=[f"{p_cost.co2_kg_m3:.0f} kg", f"{p_co2.co2_kg_m3:.0f} kg", f"{p_knee.co2_kg_m3:.0f} kg"],
                textposition="auto",
                showlegend=False
            ),
            row=1, col=2
        )
        fig_arch.add_trace(
            go.Bar(
                x=arch_names,
                y=[p_cost.fc_pred_mpa, p_co2.fc_pred_mpa, p_knee.fc_pred_mpa],
                marker_color=arch_colors,
                text=[f"{p_cost.fc_pred_mpa:.1f} MPa", f"{p_co2.fc_pred_mpa:.1f} MPa", f"{p_knee.fc_pred_mpa:.1f} MPa"],
                textposition="auto",
                showlegend=False
            ),
            row=1, col=3
        )
        fig_arch.update_layout(height=240, margin=dict(t=35, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_arch, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Export CSV Button
        csv_data = plot_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar Soluciones de Pareto en CSV (Para Laboratorio)",
            data=csv_data,
            file_name="SmartCret_Pareto_Solutions.csv",
            mime="text/csv",
        )


# =============================================================
# 4. ESCENARIOS (Visualización de Impacto de Contingencias)
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

            # VISUAL IMPACT COMPARISON CHART
            st.markdown(
                """
                <div class="chart-card">
                    <div class="chart-card-title">📊 Impacto Económico y Ambiental de la Contingencia</div>
                """,
                unsafe_allow_html=True,
            )
            fig_cont = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Costo por m³ (S/ / m³)", "Emisiones de CO₂ (kg / m³)"),
                horizontal_spacing=0.15
            )
            fig_cont.add_trace(
                go.Bar(
                    x=["Base", "Contingencia"],
                    y=[impact_res.cost_original, impact_res.cost_new],
                    marker_color=["#64748B", "#EF4444" if impact_res.cost_delta > 0 else "#34D399"],
                    text=[f"S/ {impact_res.cost_original:.1f}", f"S/ {impact_res.cost_new:.1f} ({impact_res.cost_delta_pct:+.1f}%)"],
                    textposition="auto",
                    showlegend=False
                ),
                row=1, col=1
            )
            fig_cont.add_trace(
                go.Bar(
                    x=["Base", "Contingencia"],
                    y=[impact_res.co2_original, impact_res.co2_new],
                    marker_color=["#64748B", "#F59E0B" if impact_res.co2_delta > 0 else "#34D399"],
                    text=[f"{impact_res.co2_original:.1f} kg", f"{impact_res.co2_new:.1f} kg ({impact_res.co2_delta_pct:+.1f}%)"],
                    textposition="auto",
                    showlegend=False
                ),
                row=1, col=2
            )
            fig_cont.update_layout(height=250, margin=dict(t=35, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
            st.plotly_chart(fig_cont, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

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
                if st.button("📄 Generar y Descargar Ficha Técnica Oficial en PDF (ACI 301)"):
                    pdf_path = scenario_manager.export_pdf_report(selected_scens, pdf_target)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Descargar Reporte PDF Oficial de Ingeniería",
                            data=f,
                            file_name="SmartCret_Ficha_Tecnica.pdf",
                            mime="application/pdf",
                        )


# =============================================================
# 5. MATERIALES (Bubble Chart Ecoeficiente 2D)
# =============================================================
elif selected_module == "Materiales":
    st.markdown("<div class='module-header'>🗄️ Catálogo y Gestión de Materiales (Calibrado Yura)</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Visualización de costo unitario vs. huella de carbono según Declaraciones Ambientales de Producto (EPD).</div>", unsafe_allow_html=True)

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
                "Normativa / EPD": "ASTM C595 / ISO 14067 ✅" if "yura" in k.lower() or "puzolana" in k.lower() else "ASTM C33 / C150",
                "Disponible": "Sí ✅" if v.get("available", True) else "No ❌",
            }
        )
    df_mat = pd.DataFrame(mat_rows)

    # VISUAL 2D BUBBLE CHART (Cost vs. CO2)
    st.markdown(
        """
        <div class="chart-card">
            <div class="chart-card-title">🫧 Matriz de Ecoeficiencia: Costo Unitario vs. Factor de Emisión CO₂</div>
        """,
        unsafe_allow_html=True,
    )
    fig_mat_bubble = px.scatter(
        df_mat,
        x="Factor Emisión (kg CO₂e/kg)",
        y="Costo Unitario (S/ / kg)",
        size="Densidad (kg/m³)",
        color="Factor Emisión (kg CO₂e/kg)",
        text="Nombre Comercial",
        color_continuous_scale="Viridis",
        title="Posicionamiento de Insumos: Cementos YURA y Puzolanas en la Zona de Alta Ecoeficiencia",
    )
    fig_mat_bubble.update_traces(textposition='top center')
    fig_mat_bubble.update_layout(height=360, margin=dict(t=40, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
    st.plotly_chart(fig_mat_bubble, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 Ver Tabla Detallada del Inventario de Insumos"):
        st.dataframe(df_mat, use_container_width=True, hide_index=True)


# =============================================================
# 6. PREDICCIÓN IA (Doble Tacómetro Gauge + Curva de Maduración)
# =============================================================
elif selected_module == "Predicción IA":
    st.markdown("<div class='module-header'>🧬 Laboratorio Virtual & Predicción con IA</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Inferencia en tiempo real según ASTM C39 (compresión) y ASTM C143 (slump), con bandas de incertidumbre conformal al 90%.</div>", unsafe_allow_html=True)

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

        # DUAL HIGH PRECISION GAUGES (Strength and Slump)
        st.markdown(
            """
            <div class="chart-card" style="margin-top: 10px;">
                <div class="chart-card-title">⏱️ Predicción en Tiempo Real con Intervalos Conformales (90% Confianza)</div>
            """,
            unsafe_allow_html=True,
        )
        g_col1, g_col2 = st.columns(2)

        with g_col1:
            fig_fc_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_res.resistencia_mpa,
                title = {'text': f"Resistencia f'c ASTM C39: <b>[{pred_res.resistencia_lower:.1f} - {pred_res.resistencia_upper:.1f}] MPa</b>", 'font': {'size': 13, 'color': '#F8FAFC'}},
                gauge = {
                    'axis': {'range': [10, 55], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
                    'bar': {'color': '#38BDF8'},
                    'bgcolor': 'rgba(15,23,42,0.8)',
                    'steps': [
                        {'range': [10, 21], 'color': 'rgba(148, 163, 184, 0.2)'},
                        {'range': [21, 28], 'color': 'rgba(251, 191, 36, 0.25)'},
                        {'range': [28, 55], 'color': 'rgba(52, 211, 153, 0.35)'},
                    ],
                    'threshold': {
                        'line': {'color': '#34D399', 'width': 4},
                        'thickness': 0.8,
                        'value': 28.0
                    }
                }
            ))
            fig_fc_gauge.update_layout(height=190, margin=dict(t=35, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_fc_gauge, use_container_width=True)

        with g_col2:
            fig_slump_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pred_res.slump_mm,
                title = {'text': f"Asentamiento ASTM C143: <b>[{pred_res.slump_lower:.0f} - {pred_res.slump_upper:.0f}] mm</b>", 'font': {'size': 13, 'color': '#F8FAFC'}},
                gauge = {
                    'axis': {'range': [0, 200], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
                    'bar': {'color': '#FBBF24'},
                    'bgcolor': 'rgba(15,23,42,0.8)',
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(56, 189, 248, 0.2)'},
                        {'range': [50, 100], 'color': 'rgba(52, 211, 153, 0.3)'},
                        {'range': [100, 150], 'color': 'rgba(251, 191, 36, 0.3)'},
                        {'range': [150, 200], 'color': 'rgba(239, 68, 68, 0.3)'},
                    ],
                    'threshold': {
                        'line': {'color': '#38BDF8', 'width': 4},
                        'thickness': 0.8,
                        'value': 100.0
                    }
                }
            ))
            fig_slump_gauge.update_layout(height=190, margin=dict(t=35, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_slump_gauge, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

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

        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">📈 Curva de Ganancia de Resistencia en el Tiempo (Método de Madurez ASTM C1074)</div>
            """,
            unsafe_allow_html=True,
        )
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=df_curve["Edad"], y=df_curve["Upper"], fill=None, mode='lines', line_color='rgba(56, 189, 248, 0.2)', name='Límite Superior 90%'))
        fig_curve.add_trace(go.Scatter(x=df_curve["Edad"], y=df_curve["Lower"], fill='tonexty', mode='lines', line_color='rgba(56, 189, 248, 0.2)', name='Banda de Confianza Conformal 90%'))
        fig_curve.add_trace(go.Scatter(x=df_curve["Edad"], y=df_curve["Resistencia"], mode='lines+markers', line=dict(color='#38BDF8', width=3), name='Resistencia Estimada'))
        fig_curve.update_layout(xaxis_title="Edad de curado (días)", yaxis_title="Resistencia a compresión (MPa)", height=300, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)")
        st.plotly_chart(fig_curve, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with p_tab2:
        render_interactive_assistant(key_prefix="pred_tab")


# =============================================================
# 7. HUELLA DE CARBONO (Monitoreo Semafórico y Distribución)
# =============================================================
elif selected_module == "Asistente IA":
    render_interactive_assistant(key_prefix="main_agent")

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
        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">⏱️ Semáforo de Consumo de Presupuesto de CO₂</div>
            """,
            unsafe_allow_html=True,
        )
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = co2_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Emisiones Acumuladas vs. Presupuesto (t CO₂)", 'font': {'size': 16, 'color': '#F8FAFC'}},
            delta = {'reference': co2_max, 'increasing': {'color': "#EF4444"}},
            gauge = {
                'axis': {'range': [None, co2_max * 1.25], 'tickwidth': 1, 'tickcolor': '#94A3B8'},
                'bar': {'color': "#38BDF8"},
                'bgcolor': "rgba(30,41,59,0.8)",
                'steps': [
                    {'range': [0, co2_max * 0.8], 'color': "rgba(52,211,153,0.3)"},
                    {'range': [co2_max * 0.8, co2_max], 'color': "rgba(251,191,36,0.35)"},
                    {'range': [co2_max, co2_max * 1.25], 'color': "rgba(239,68,68,0.4)"},
                ],
                'threshold': {
                    'line': {'color': "#EF4444", 'width': 4},
                    'thickness': 0.75,
                    'value': co2_max,
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(t=30, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with h_col2:
        st.markdown(
            """
            <div class="chart-card">
                <div class="chart-card-title">🌱 Balance Ambiental de Obra</div>
            """,
            unsafe_allow_html=True,
        )
        co2_evitado = (status['poured_volume_m3'] * 325.0 / 1000.0) - co2_val
        st.markdown(
            f"""
            <div style="font-size: 0.9rem; color: #CBD5E1; line-height: 1.8;">
                • Obra Activa: <b style="color:#F8FAFC;">{status['project_name']}</b><br/>
                • Volumen Vaciado: <b>{status['poured_volume_m3']} m³</b> / {status['planned_volume_m3']} m³<br/>
                • Cuota Consumida: <b style="color:#38BDF8;">{co2_pct:.1f}%</b> ({co2_val:.1f} t CO₂)<br/>
                • Saldo Autorizado: <b>{status['co2_remaining_ton']:.1f} t CO₂</b>
            </div>
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 8px; padding: 12px; margin-top: 14px; text-align: center;">
                <span style="font-size: 1.3rem;">🌳</span><br/>
                <b style="color: #34D399; font-size: 1.1rem;">{co2_evitado:.1f} Toneladas de CO₂ Evitadas</b><br/>
                <span style="font-size: 0.75rem; color: #94A3B8;">Equivalente a ~{int(co2_evitado * 45)} árboles plantados</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# 8. ALERTAS
# =============================================================
elif selected_module == "Alertas":
    st.markdown("<div class='module-header'>🔔 Centro de Alertas y Notificaciones</div>", unsafe_allow_html=True)
    st.markdown("<div class='module-sub'>Monitoreo activo de límites presupuestales, normativos ACI y disponibilidad de insumos.</div>", unsafe_allow_html=True)

    alerts = carbon_tracker.check_alerts("OBRA-LIMA-01")

    # Add simulated system alerts for demonstration
    all_alerts = [
        {"Nivel": "CRÍTICA", "Icono": "🔴", "Tipo": "Presupuesto", "Mensaje": "Alerta preventiva: Se ha alcanzado el 82.5% de la cuota de carbono autorizada para la Obra Las Palmeras.", "Hora": "Hace 10 min"},
        {"Nivel": "PREVENTIVA", "Icono": "🟡", "Tipo": "Materiales", "Mensaje": "Stock de Escoria GGBS por debajo del umbral de reserva estratégica (quedan 18 toneladas).", "Hora": "Hace 2 horas"},
        {"Nivel": "INFORMATIVA", "Icono": "🟢", "Tipo": "Conformidad", "Mensaje": "Mezcla certificada: El diseño f'c = 28 MPa cumple con los requisitos estructurales y de durabilidad del código ACI 318 con 90% de confianza estadística.", "Hora": "Hoy 08:30 AM"},
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
