# Supply Chain Optimization with Graph ML & RL

A production-grade supply chain optimization system that uses **Heterogeneous Graph Transformers (HGT)** and **Proximal Policy Optimization (PPO)** to solve dynamic routing and vehicle selection problems across a realistic logistics network of Indian cities.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Scenarios](#scenarios)
8. [Testing](#testing)
9. [Dashboard](#dashboard)
10. [Deployment (Cloud Run)](#deployment-cloud-run)
11. [Configuration](#configuration)
12. [Roadmap](#roadmap)

---

## Overview

Traditional supply chain routing relies on static heuristics that break under real-world disruptions (weather, traffic, geopolitical events). This project replaces them with a **GNN-RL agent** that:

- Encodes the entire supply chain as a **heterogeneous graph** (cities, warehouses, vehicles, shipments).
- Processes graph state at every step with an **HGT encoder** using multi-head attention.
- Makes simultaneous decisions for **next-hop** and **vehicle mode** via a Pointer Network policy.
- Is trained end-to-end with **PPO** using a 3-phase curriculum that progressively increases disruption severity.
- Streams live simulation state to a **WebSocket dashboard** built on FastAPI and Leaflet.js.

**Key results:**
| Agent | Success Rate | Notes |
|---|---|---|
| Random | ~0% | No sense of direction or risk |
| Greedy (shortest path) | ~40% | Fails under volatility |
| Trained GNN+RL | **~90%** | Learns network risk dynamics |

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                  SupplyChainEnv (Gymnasium)            │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│   │ AnomalyEngine│  │  TimeEngine  │  │CostCalculator│ │
│   └─────────────┘  └──────────────┘  └─────────────┘  │
│               ↓ get_graph_state()                     │
└───────────────────────────────────────────────────────┘
                     │
              FeatureEngine
          (env state → HeteroData)
                     │
           ┌─────────▼──────────┐
           │    GNNEncoder      │
           │   (HGT 2-layer)    │
           │  node & graph embs │
           └─────────┬──────────┘
                     │
           ┌─────────▼──────────┐
           │    ActorCritic     │
           │  Pointer Network   │
           │  next-hop + vehicle│
           │  Critic Value head │
           └─────────┬──────────┘
                     │
              PPO Training Loop
           (GAE, clipped surrogate,
            curriculum scheduler)
```

### Graph Schema

| Entity | Node/Edge | Key Features |
|--------|-----------|--------------|
| **Location** | Node | lat/lng, region type, warehouse fill ratio, cold storage, handling cost, on-nominal-path flag |
| **Vehicle** | Node | type (truck/rail/air/ship), payload, fuel efficiency, speed, maintenance cost |
| **Shipment** | Node | product type, fragility, shelf life, weight, value, priority |
| **route** | Edge (Location→Location) | distance, terrain, road grading, toll cost, anomaly modifiers |
| **vehicle_at** | Edge (Vehicle→Location) | — |
| **shipment_at** | Edge (Shipment→Location) | — |
| **shipment_dest** | Edge (Shipment→Location) | — |

---

## Project Structure

```
.
├── main.py                   # CLI entry point (train / eval / random / dashboard)
├── pyproject.toml            # uv project & dependency manifest
├── Dockerfile                # Production image for Cloud Run
├── src/
│   ├── evaluate.py           # Multi-policy evaluation suite
│   ├── config/
│   │   ├── default_config.py # Dataclasses: LocationConfig, RouteConfig, VehicleConfig, …
│   │   └── scenarios.py      # small_scenario(), india_scenario(), volatile_scenario()
│   ├── environment/
│   │   ├── supply_chain_env.py  # SupplyChainEnv (Gymnasium)
│   │   ├── anomaly_engine.py    # Stochastic disruption injection
│   │   ├── cost_calculator.py   # Per-leg monetary cost breakdown
│   │   └── time_engine.py       # Traffic, seasonality, holidays
│   ├── features/
│   │   └── feature_engine.py    # env state dict → torch_geometric HeteroData
│   ├── models/
│   │   ├── gnn_encoder.py       # HGT-based GNN encoder
│   │   ├── ppo_agent.py         # ActorCritic with Pointer Network heads
│   │   └── train.py             # PPO loop, GAE, curriculum scheduler
│   └── utils/
│       └── graph_utils.py       # Shortest-path helpers
├── dashboard/
│   ├── app.py                   # FastAPI server + WebSocket endpoint
│   └── static/                  # HTML, CSS, JS (Leaflet.js)
├── checkpoints/                 # Saved model weights (.pt)
├── tests/
│   ├── test_environment.py      # Smoke tests for env reset/step
│   ├── test_gnn.py              # GNN forward-pass shape checks
│   ├── test_ppo.py              # ActorCritic action selection & log-prob tests
│   └── test_reroute.py          # Controlled rerouting unit test
├── docs/
│   ├── doc.md                   # System overview & component reference
│   ├── simulation_details.md    # Environment MDP mechanics
│   ├── architecture.md          # Deep-dive: GNN, PPO, curriculum
│   ├── api_reference.md         # Module-level API reference
│   └── DEMO.md                  # 3-minute demo script
└── research/                 # Archived experiments and smaller attempts
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.13+ |
| [uv](https://github.com/astral-sh/uv) | latest |

> PyTorch, PyTorch Geometric, Gymnasium, FastAPI, and all other dependencies are managed by `uv` and declared in `pyproject.toml`.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/godofwar1007/Supply-chain-optimization-using-Graph-ML-and-RL.git
cd Supply-chain-optimization-using-Graph-ML-and-RL

# 2. Install all dependencies (creates an isolated virtual environment)
uv sync

# 3. (Optional) install dev dependencies for running tests
uv sync --group dev
```

---

## Usage

All operations go through `main.py`:

```
python main.py <command>

Commands:
  train      Train the PPO agent with 3-phase curriculum (5 000 episodes)
  eval       Run one evaluation episode with the best saved checkpoint
  random     Run one episode with a random-action baseline
  dashboard  Launch the FastAPI visualization server on port 8000
```

### Train

```bash
python main.py train
```

Training artefacts are written to `checkpoints/`:

| File | Description |
|---|---|
| `best_model.pt` | Highest avg-reward checkpoint |
| `latest_model.pt` | Most recent checkpoint (saved every N episodes) |
| `final_model.pt` | Model at end of training |
| `training_metrics.csv` | Per-episode reward, delivery rate, loss |
| `training_curves.png` | Reward / delivery / loss plots |

### Evaluate

```bash
python main.py eval     # uses best_model.pt (falls back to random if missing)
python main.py random   # random baseline for comparison
```

### Dashboard

```bash
python main.py dashboard
```

Open `http://localhost:8000` in your browser. The dashboard lets you choose a scenario (India, Volatile, Small), select an agent (Random or Trained GNN+RL), and watch the simulation unfold step-by-step on an interactive map.

---

## Scenarios

| Scenario (`config.name`) | Factory function | Nodes | Description |
|---|---|---|---|
| `small_test` | `small_scenario()` | 4 | Minimal graph for unit tests and rapid iteration |
| `india_large` | `india_scenario()` | 40 | Full Indian logistics network — major metros, secondary cities, port hubs |
| `india_volatile` | `volatile_scenario()` | 40 | Same as India but "Golden Quadrilateral" routes are fast yet highly volatile |
| `reroute_test` | `reroute_test_scenario()` | 40 | India map with stochastic anomalies disabled; used for controlled rerouting demos |

Scenarios are defined in `src/config/scenarios.py` and can be customised by modifying or subclassing `ScenarioConfig`.

---

## Testing

```bash
# Run the full test suite
pytest

# Run individual test files
python tests/test_environment.py   # Env smoke test (reset, step, rewards)
python tests/test_gnn.py           # GNN encoder shape tests
python tests/test_ppo.py           # ActorCritic action & log-prob tests
```

Test coverage:
- **`test_environment.py`** — Verifies env `reset()` / `step()` on both `small_scenario()` and `india_scenario()`, checks observation shapes and reward ranges.
- **`test_gnn.py`** — Builds a minimal `HeteroData` object and verifies `GNNEncoder` output embedding shapes for all node types.
- **`test_ppo.py`** — Verifies `ActorCritic.forward()` for action sampling (no action given) and log-probability evaluation (action given).

---

## Dashboard

The real-time dashboard is powered by **FastAPI** (backend) and **Leaflet.js** (frontend).

### Features

- **Interactive map** — CartoDB dark-tile base with nodes (cities) and edges (routes) rendered as overlays.
- **Disruption layer** — Edges/nodes highlighted by active anomaly type (weather, traffic, sentiment, geopolitical).
- **Optimal path baselines** — Nominal shortest path (dashed green) and dynamic optimal path from current position (dashed blue).
- **Live metrics** — Steps, cumulative time, cost (₹), and cargo risk updated via WebSocket after every step.
- **Step log** — Detailed breakdown of every routing decision including vehicle chosen, leg cost, and reward.
- **Agent selector** — Switch between Random Baseline and Trained GNN+RL without restarting the server.

### WebSocket Protocol

The dashboard WebSocket endpoint is `ws://localhost:8000/ws`. The server streams JSON messages of the form:

```json
{
  "type": "step",
  "step": 3,
  "current_node": "Nagpur",
  "action": [2, 0],
  "reward": -0.42,
  "done": false,
  "info": { ... }
}
```

---

## Deployment (Cloud Run)

The project ships a `Dockerfile` optimised for **Google Cloud Run**.

```bash
# Build the image
docker build -t supply-chain-agent .

# Run locally
docker run -p 8080:8080 supply-chain-agent

# Deploy to Cloud Run
gcloud run deploy supply-chain-agent \
  --image gcr.io/<PROJECT>/supply-chain-agent \
  --platform managed \
  --allow-unauthenticated \
  --no-cpu-throttling \
  --port 8080
```

Key notes:
- The `PORT` environment variable is respected; Cloud Run injects it automatically.
- Use `--no-cpu-throttling` so the simulation loop stays responsive during active WebSocket sessions.
- Dev-only dependencies (pytest) are excluded from the production image via `uv sync --no-dev`.

---

## Configuration

The entire scenario is controlled by dataclasses in `src/config/default_config.py`:

| Class | Purpose |
|---|---|
| `LocationConfig` | City node — coordinates, region type, warehouse capacity |
| `RouteConfig` | Edge — distance, terrain, road grading, toll, volatility flag |
| `VehicleConfig` | Vehicle — type, payload, speed, fuel efficiency, home hub |
| `ShipmentTemplate` | Cargo — fragility, shelf life, weight, priority, insurance value |
| `AnomalyConfig` | Per-type disruption probability and severity ranges |
| `RewardWeights` | Weights for time, cost, risk, spoilage, and delay penalties |
| `ScenarioConfig` | Top-level container — assembles all of the above |

---

## Roadmap

### ✅ Completed
- Heterogeneous graph environment with 40 Indian cities
- Stochastic anomaly engine (weather, traffic, sentiment, geopolitical)
- HGT + PPO pipeline with Pointer Network action heads
- 3-phase curriculum learning with automatic phase transitions
- Real-time WebSocket dashboard with Leaflet.js map
- Optimal path baselines (static and dynamic)
- Docker + Cloud Run deployment