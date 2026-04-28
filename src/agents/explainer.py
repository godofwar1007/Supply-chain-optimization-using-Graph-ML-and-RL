"""
Vertex AI Path Deviation Explainer Agent.

Uses the Google Gen AI SDK with Gemini 3 Flash to generate natural-language
explanations whenever the RL agent deviates from the known optimal
(Dijkstra-based) path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-loaded client to avoid import errors when the SDK isn't installed
_client = None
_available: bool | None = None

MODEL_ID = "gemini-2.5-flash"

SYSTEM_PROMPT = """\
You are Data2Delivery's logistics intelligence system. Your role is to analyze
path decisions made by a reinforcement-learning routing agent and explain WHY
the agent chose a different route than the nominal shortest path.

You are given structured context about the current simulation state:
- The optimal next hop (shortest path by base travel time, ignoring anomalies)
- The agent's actual chosen next hop
- Active network disruptions (weather, traffic, geopolitical, sentiment)
- Route metrics (cost, time, risk) for both options
- Shipment constraints (shelf life remaining, priority, weight)
- The agent's journey history

Provide a concise 2-3 sentence explanation that a supply chain manager would
understand. Focus on the most likely reason: anomalies on the optimal route,
risk avoidance, shelf-life pressure, or cost optimization. Be specific about
which disruptions or metrics likely influenced the decision. Do not speculate
beyond the provided data.\
"""


def _get_client():
    """Lazily initialize the Google Gen AI client for Vertex AI."""
    global _client, _available
    if _client is not None:
        return _client

    try:
        from google import genai

        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "data2delivery")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-south1")

        _client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )
        _available = True
        logger.info(
            "Vertex AI Explainer initialized (project=%s, location=%s, model=%s)",
            project,
            location,
            MODEL_ID,
        )
        return _client

    except Exception as e:
        _available = False
        logger.warning("Vertex AI Explainer unavailable: %s", e)
        return None


def is_available() -> bool:
    """Check whether the Vertex AI explainer can be used."""
    global _available
    if _available is None:
        _get_client()
    return _available or False


def _build_prompt(context: dict[str, Any]) -> str:
    """Build a structured user prompt from the simulation context."""
    lines = [
        f"## Current Decision (Step {context.get('step', '?')})",
        f"- Agent is at: **{context.get('current_node', '?')}**",
        f"- Destination: **{context.get('destination', '?')}**",
        f"- Optimal next hop (Dijkstra): **{context.get('optimal_next_hop', '?')}**",
        f"- Agent's chosen hop: **{context.get('chosen_hop', '?')}**",
        f"- Vehicle used: **{context.get('vehicle_type', '?')}**",
        "",
        "## Route Metrics for This Leg",
        f"- Travel time: {context.get('time_hours', '?')} hours",
        f"- Cost: ₹{context.get('cost', '?')}",
        f"- Risk score: {context.get('risk', '?')}",
        "",
        "## Shipment Constraints",
        f"- Shelf life remaining: {context.get('shelf_remaining_pct', '?')}%",
        f"- Priority: {context.get('priority', '?')}",
        f"- Product type: {context.get('product_type', '?')}",
        "",
        "## Cumulative Journey So Far",
        f"- Total time: {context.get('total_time', '?')} hours",
        f"- Total cost: ₹{context.get('total_cost', '?')}",
        f"- Total risk: {context.get('total_risk', '?')}",
        f"- Path history: {' → '.join(context.get('path_history', []))}",
        "",
    ]

    # Active anomalies on the optimal edge
    optimal_anomalies = context.get("optimal_edge_anomalies", [])
    if optimal_anomalies:
        lines.append("## Disruptions on the OPTIMAL Route")
        for a in optimal_anomalies:
            lines.append(
                f"- **{a.get('type', '?')}** — severity {a.get('severity', '?')}x"
            )
        lines.append("")

    # Active anomalies on the chosen edge
    chosen_anomalies = context.get("chosen_edge_anomalies", [])
    if chosen_anomalies:
        lines.append("## Disruptions on the CHOSEN Route")
        for a in chosen_anomalies:
            lines.append(
                f"- **{a.get('type', '?')}** — severity {a.get('severity', '?')}x"
            )
        lines.append("")

    # Global anomalies summary
    global_count = context.get("total_active_anomalies", 0)
    lines.append(f"## Network State: {global_count} total active disruptions")
    lines.append("")
    lines.append(
        "Explain why the RL agent likely chose this different route instead of "
        "the optimal path."
    )

    return "\n".join(lines)


def explain_deviation(context: dict[str, Any]) -> str:
    """
    Generate an AI explanation for why the agent deviated from the optimal path.

    This is a synchronous function (the Gen AI SDK's generate_content is sync).
    Call via ``asyncio.to_thread(explain_deviation, ctx)`` in async contexts.

    Parameters
    ----------
    context : dict
        Structured context about the current simulation state.

    Returns
    -------
    str
        A natural-language explanation, or empty string on failure.
    """
    client = _get_client()
    if client is None:
        return ""

    prompt = _build_prompt(context)

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=200,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
            ),
        )
        return response.text or ""

    except Exception as e:
        logger.error("Explainer API call failed: %s", e)
        return ""

