import re
from pathlib import Path

app_path = Path("src/smartcret/app/app.py")
content = app_path.read_text(encoding="utf-8")

# 1. Update menu_options and icons
old_menu = '''    menu_options = [
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
    }'''

new_menu = '''    menu_options = [
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
    }'''

if old_menu in content:
    content = content.replace(old_menu, new_menu)
    print("Menu options successfully updated with 'Asistente IA'!")
else:
    print("Warning: old_menu not found directly, checking...")

# 2. Add the interactive assistant function definition before the page router
assistant_func_code = '''
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
                    "👋 **¡Hola! Soy el Asistente Agéntico de Ingeniería de SmartCret.**\\n\\n"
                    "Estoy conectado a los motores de **Optimización Multiobjetivo (NSGA-II)**, "
                    "**Predicción Conformal ML (ASTM C39 / C143)** y **Simulación de Contingencias** bajo normativa "
                    "**ACI 318-19 / ACI 211.1** con la base de datos certificada de **Cementos YURA S.A.**\\n\\n"
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
            resp = assistant.process_query(query_to_process)
            st.session_state["agent_messages"].append({
                "role": "assistant",
                "content": resp["response"],
                "tools": resp["tools_called"],
                "raw": resp["raw_tool_results"],
                "suggestions": resp.get("suggestions", []),
            })
        st.rerun()
'''

# Insert render_interactive_assistant before '# 1. PANEL PRINCIPAL'
panel_mark = "# 1. PANEL PRINCIPAL"
if panel_mark in content:
    content = content.replace(panel_mark, assistant_func_code + "\n\n" + panel_mark)
    print("render_interactive_assistant inserted successfully!")
else:
    print("Error: panel_mark not found!")

# 3. Update p_tab2 inside Predicción IA
old_tab2 = '''    with p_tab2:
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
                    st.json(resp["raw_tool_results"])'''

new_tab2 = '''    with p_tab2:
        render_interactive_assistant(key_prefix="pred_tab")'''

if old_tab2 in content:
    content = content.replace(old_tab2, new_tab2)
    print("p_tab2 updated to call render_interactive_assistant!")
else:
    print("Warning: old_tab2 not found directly!")

# 4. Add 'elif selected_module == "Asistente IA":' before 'elif selected_module == "Huella de carbono":'
carbon_mark = 'elif selected_module == "Huella de carbono":'
asistente_block = '''elif selected_module == "Asistente IA":
    render_interactive_assistant(key_prefix="main_agent")

'''

if carbon_mark in content:
    content = content.replace(carbon_mark, asistente_block + carbon_mark)
    print("Asistente IA dedicated page added successfully!")
else:
    print("Error: carbon_mark not found!")

app_path.write_text(content, encoding="utf-8")
print("app.py updated successfully!")
