"""
Dashboard Backend — FastAPI server for the supply chain visualization.

Provides:
  GET  /api/network     — full graph topology (nodes + edges)
  GET  /api/model-status — check if a trained model is available
  POST /api/simulate    — run a single episode and return step-by-step trace
  WS   /ws/simulate     — stream simulation steps in real-time over WebSocket
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import random
from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.scenarios import india_scenario, small_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.ppo_agent import ActorCritic
from src.utils.graph_utils import get_neighbors


app = FastAPI(title="Supply Chain Optimizer — Dashboard")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Pre-build scenario config
SCENARIOS = {
    "india": india_scenario,
    "small": small_scenario,
}

# ═══════════════════════════════════════════════════════════════════════
# Trained Agent Loader
# ═══════════════════════════════════════════════════════════════════════

CKPT_DIR = ROOT / "checkpoints"
_loaded_agent: Optional[ActorCritic] = None
_feature_engine: Optional[FeatureEngine] = None
_agent_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_path() -> Optional[Path]:
    """Find the best available model checkpoint."""
    for name in ["best_model.pt", "final_model.pt", "latest_model.pt"]:
        p = CKPT_DIR / name
        if p.exists():
            return p
    return None


def _load_agent(scenario_name: str = "india") -> Optional[ActorCritic]:
    """Load the trained agent from checkpoint. Returns None if unavailable."""
    global _loaded_agent, _feature_engine

    model_path = _get_model_path()
    if model_path is None:
        return None

    if _loaded_agent is not None:
        return _loaded_agent

    try:
        config = SCENARIOS[scenario_name]()
        env = SupplyChainEnv(config, render_mode=None)
        _feature_engine = FeatureEngine()

        # Must match the exact metadata used during training (src/models/train.py)
        training_metadata = (
            ['location', 'vehicle', 'shipment'],
            [('location', 'route', 'location'),
             ('vehicle', 'vehicle_at', 'location'),
             ('shipment', 'shipment_at', 'location'),
             ('shipment', 'shipment_dest', 'location'),
             ('location', 'rev_vehicle_at', 'vehicle'),
             ('location', 'rev_shipment_at', 'shipment'),
             ('location', 'rev_shipment_dest', 'shipment')]
        )

        agent = ActorCritic(
            metadata=training_metadata,
            hidden_channels=64,
            out_channels=64,
            max_neighbors=env.max_neighbors,
            max_vehicles=env.max_vehicles,
        ).to(_agent_device)

        checkpoint = torch.load(model_path, map_location=_agent_device, weights_only=False)
        agent.load_state_dict(checkpoint["model_state_dict"])
        agent.eval()

        _loaded_agent = agent
        print(f"Loaded trained agent from {model_path.name} (ep {checkpoint.get('episode', '?')})")
        return agent

    except Exception as e:
        print(f"Failed to load agent: {e}")
        return None


def _agent_select_action(agent: ActorCritic, env: SupplyChainEnv, feature_engine: FeatureEngine):
    """Use the trained agent to select an action."""
    state_dict = env.get_graph_state()
    hetero_data = feature_engine.build(state_dict).to(_agent_device)

    with torch.no_grad():
        action, _, _, _ = agent(hetero_data)

    return action.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════

class SimulateRequest(BaseModel):
    scenario: str = "india"
    seed: int | None = None
    speed_ms: int = 800


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_network_json(config) -> dict:
    """Convert scenario config into a JSON-serializable network description."""
    nodes = []
    for loc in config.locations:
        nodes.append({
            "id": loc.name,
            "lat": loc.lat,
            "lng": loc.lng,
            "region_type": loc.region_type,
            "has_warehouse": loc.has_warehouse,
            "warehouse_capacity": loc.warehouse_capacity,
            "fill_ratio": loc.warehouse_fill_ratio,
            "has_cold_storage": loc.cold_storage,
        })

    edges = []
    for route in config.routes:
        src_loc = config.location_by_name(route.source)
        tgt_loc = config.location_by_name(route.target)
        edges.append({
            "source": route.source,
            "target": route.target,
            "distance_km": route.distance_km,
            "terrain": route.terrain,
            "road_grading": route.road_grading,
            "toll_cost": route.toll_cost,
            "source_lat": src_loc.lat,
            "source_lng": src_loc.lng,
            "target_lat": tgt_loc.lat,
            "target_lng": tgt_loc.lng,
        })
        if route.bidirectional:
            edges.append({
                "source": route.target,
                "target": route.source,
                "distance_km": route.distance_km,
                "terrain": route.terrain,
                "road_grading": route.road_grading,
                "toll_cost": route.toll_cost,
                "source_lat": tgt_loc.lat,
                "source_lng": tgt_loc.lng,
                "target_lat": src_loc.lat,
                "target_lng": src_loc.lng,
            })

    vehicles = []
    for v in config.vehicles:
        vehicles.append({
            "id": v.vehicle_id,
            "type": v.vehicle_type,
            "max_payload_kg": v.max_payload_kg,
            "speed_kmph": v.speed_kmph,
            "home_location": v.home_location,
        })

    return {"nodes": nodes, "edges": edges, "vehicles": vehicles}


# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Serve the main dashboard page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/network")
async def get_network(scenario: str = "india"):
    """Return the full supply chain network topology."""
    config = SCENARIOS.get(scenario, SCENARIOS["india"])()
    return _build_network_json(config)


@app.get("/api/model-status")
async def model_status():
    """Check if a trained model is available."""
    model_path = _get_model_path()
    if model_path is None:
        return {"available": False, "model": None}

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return {
        "available": True,
        "model": model_path.name,
        "episode": checkpoint.get("episode", "?"),
    }


@app.websocket("/ws/simulate")
async def websocket_simulate(websocket: WebSocket):
    """
    Stream simulation steps over WebSocket for real-time visualization.

    Client sends: {"scenario": "india", "seed": 42, "speed_ms": 800, "agent": "trained"}
    Server streams: one JSON message per step, then a final summary.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        scenario_name = data.get("scenario", "india")
        seed = data.get("seed") or random.randint(0, 100000)
        speed_ms = data.get("speed_ms", 800)
        agent_mode = data.get("agent", "random")  # "random" or "trained"

        config = SCENARIOS.get(scenario_name, SCENARIOS["india"])()
        env = SupplyChainEnv(config, render_mode=None)
        obs, info = env.reset(seed=seed)

        # Load trained agent if requested
        agent = None
        feature_engine = None
        if agent_mode == "trained":
            agent = _load_agent(scenario_name)
            if agent is not None:
                feature_engine = _feature_engine or FeatureEngine()
            else:
                agent_mode = "random"  # Fallback

        # Send initial state
        await websocket.send_json({
            "type": "init",
            "source": env.path_taken[0],
            "destination": env.destination,
            "agent_mode": agent_mode,
            "shipment": {
                "product_type": env.shipment.product_type,
                "weight_kg": round(env.shipment.weight_kg, 0),
                "priority": env.shipment.priority,
                "shelf_life_hours": env.shipment.shelf_life_hours,
            },
            "network": _build_network_json(config),
        })

        done = False
        truncated = False

        while not (done or truncated):
            # Select action based on mode
            if agent_mode == "trained" and agent is not None:
                action = _agent_select_action(agent, env, feature_engine)
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)

            leg = env.leg_details[-1]
            anomalies = [
                {"type": a["type"], "severity": round(a["severity"], 2)}
                for a in leg.get("anomalies", [])
            ]

            src_loc = config.location_by_name(leg["from"])
            tgt_loc = config.location_by_name(leg["to"])

            await websocket.send_json({
                "type": "step",
                "step": env.step_count,
                "from": leg["from"],
                "to": leg["to"],
                "from_lat": src_loc.lat,
                "from_lng": src_loc.lng,
                "to_lat": tgt_loc.lat,
                "to_lng": tgt_loc.lng,
                "vehicle_type": leg["vehicle_type"],
                "time_hours": round(leg["time_hours"], 1),
                "cost": round(leg["cost"].total, 0),
                "risk": round(leg["risk"], 3),
                "anomalies": anomalies,
                "reward": round(reward, 2),
                "total_time": round(env.total_time_hours, 1),
                "total_cost": round(env.total_cost, 0),
                "total_risk": round(env.total_risk, 3),
                "delivered": env.current_node == env.destination,
                "shelf_remaining_pct": round(
                    max(0, 100 * (1 - env.total_time_hours / env.shipment.shelf_life_hours)), 1
                ),
            })

            await asyncio.sleep(speed_ms / 1000.0)

        # Send summary
        await websocket.send_json({
            "type": "done",
            "delivered": env.current_node == env.destination,
            "path": env.path_taken,
            "total_steps": env.step_count,
            "total_time_hours": round(env.total_time_hours, 1),
            "total_cost": round(env.total_cost, 0),
            "total_risk": round(env.total_risk, 3),
        })

    except WebSocketDisconnect:
        pass
