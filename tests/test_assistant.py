"""Unit tests for Phase 5: Agentic Assistant and verified tool execution."""

import pytest
from smartcret.assistant.assistant import SmartCretAssistant


@pytest.fixture(scope="module")
def assistant():
    return SmartCretAssistant()


def test_assistant_contingency_query_acceptance(assistant):
    """Criterio de Aceptación Fase 5:

    Responde '¿qué mezcla me conviene si sube el precio del cemento?'
    llamando a las herramientas y explicando el resultado.
    """
    user_query = "¿qué mezcla me conviene si sube el precio del cemento?"
    result = assistant.process_query(user_query)

    assert "tool_simulate_contingency" in result["tools_called"]
    response = result["response"]

    # Must contain engineering details and numbers from tool execution
    assert "Cemento:" in response
    assert "kg/m³" in response
    assert "MPa" in response
    assert "S/" in response
    assert "Intervalo 90%" in response
    assert len(result["raw_tool_results"]) > 0


def test_assistant_optimization_query(assistant):
    user_query = "optimiza una mezcla para 30 MPa y slump 100 mm"
    result = assistant.process_query(user_query)

    assert "tool_optimize_mix" in result["tools_called"]
    assert "Frente de Pareto" in result["response"]
    assert "Mínimo Costo" in result["response"]
