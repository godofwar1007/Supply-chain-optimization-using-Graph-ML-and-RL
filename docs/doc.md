# Supply Chain Optimization — System Documentation

## Overview

This project implements a production-grade supply chain optimization system using **Heterogeneous Graph Machine Learning** and **Reinforcement Learning**. It replaces traditional fixed-route heuristics with an intelligent agent capable of dynamic routing and vehicle selection in a volatile environment.

### Key Capabilities
- **Dynamic Routing**: Real-time pathfinding on a heterogeneous graph updated at every environment step.
- **Anomaly Awareness**: Native handling of four stochastic disruption types: weather, traffic, social sentiment, and geopolitical events.
- **Multi-Objective Optimization**: Simultaneous minimization of travel time, monetary cost, and cargo risk.
- **Curriculum Learning**: Automatic 3-phase training progression that ramps anomaly difficulty as the agent improves.
- **Real-Time Visualization**: WebSocket-based dashboard built on FastAPI and Leaflet.js.

---

## 1. Graph Architecture

The supply chain is modeled as a **heterogeneous graph** (`torch_geometric.data.HeteroData`) where different entity types carry specialized feature vectors.

### 1.1 Node Types

| Node Type | Config Class | Key Features |
|-----------|-------------|--------------|
| **Location** | `LocationConfig` | lat, lng, region type (metro/urban/hub/port/rural/coastal), warehouse fill ratio, cold storage flag, handling cost, throughput rate, on-nominal-path flag |
| **Vehicle** | `VehicleConfig` | type (truck/rail/air/ship), max payload (kg), fuel efficiency (km/L), age (years), maintenance cost/km, base speed (km/h), home location |
| **Shipment** | `ShipmentTemplate` | product type, fragility (0–1), shelf life (hours), temperature sensitivity, weight (kg), volume (m³), priority, insurance value (₹) |

> Warehouse metadata is embedded directly into `Location` node features rather than as a separate node type.

### 1.2 Edge Types

| Edge | Direction | Meaning |
|------|-----------|---------|
| `route` | Location → Location | Bidirectional transport link; carries distance, terrain, road grading, toll cost, and active anomaly modifiers |
| `vehicle_at` | Vehicle → Location | Vehicle is currently at this city (+ reverse `rev_vehicle_at`) |
| `shipment_at` | Shipment → Location | Shipment is currently at this city (+ reverse `rev_shipment_at`) |
| `shipment_dest` | Shipment → Location | Shipment's destination city (+ reverse `rev_shipment_dest`) |

### 1.3 Global Context

Each `HeteroData` object also carries two extra tensors (not node features):

- **`global_context`** `[1, 10]` — sine/cosine cyclical encodings of hour, day, month; monsoon flag, holiday flag, fuel price multiplier.
- **`step_progress`** `[1, 2]` — current step fraction and remaining shelf-life fraction, allowing the agent to sense urgency.

---

## 2. Feature Engine (`src/features/feature_engine.py`)

`FeatureEngine.build(state_dict)` converts the raw Python dict returned by `env.get_graph_state()` into a `HeteroData` object ready for the GNN.

**Key encoding decisions:**

| Raw value | Encoding |
|-----------|---------|
| Region type (metro/urban/…) | Float scalar: metro=1.0, urban=0.7, hub=0.8, port=0.6, coastal=0.5, rural=0.3 |
| Terrain type | Float scalar: flat=0.2, coastal=0.4, hilly=0.5, mountainous=0.8 |
| Vehicle type | Float scalar: truck=0.25, rail=0.5, air=0.75, ship=1.0 |
| Priority | Float scalar: low=0.25, medium=0.5, high=0.75, critical=1.0 |
| `on_nominal_path` | Binary 0/1 injected into location feature vector |

The output `HeteroData` also stores `neighbor_indices` and `current_node_idx` as graph-level attributes used by the actor heads.

---

## 3. GNN Encoder (`src/models/gnn_encoder.py`)

`GNNEncoder` processes a `HeteroData` object and returns per-node embeddings plus a graph-level embedding.

### Architecture

```
Raw node features (variable dims per type)
        │
Type-specific Linear projections  →  hidden_channels (default: 64)
        │
HGTConv layer × num_layers (default: 2)   [multi-head attention, 4 heads]
        │
Output Linear projection  →  out_channels (default: 64)
        │
  node_embeddings: {location: Tensor, vehicle: Tensor, shipment: Tensor}
        │
Mean-pool location + shipment  +  context_mlp(global_context ∥ step_progress)
        │
  graph_embedding: Tensor [1, out_channels]
```

**Outputs:**
- `node_embeddings["location"]` — `[num_locations, 64]`
- `node_embeddings["vehicle"]` — `[num_vehicles, 64]`
- `node_embeddings["shipment"]` — `[num_shipments, 64]`
- `graph_embedding` — `[1, 64]`, used by the Critic and as the query basis for Actor heads

---

## 4. Actor-Critic Agent (`src/models/ppo_agent.py`)

`ActorCritic` wraps the GNN encoder with a Critic value head and two Pointer Network Actor heads.

### Critic

```
graph_embedding  →  Linear(64→64)  →  ReLU  →  Linear(64→1)  →  value scalar
```

### Actor — Pointer Network

Both the neighbor head and the vehicle head follow the same dot-product attention pattern:

```
query = MLP([graph_embedding ∥ current_node_embedding])   [1, 64]
key   = Linear(candidate_embedding)                        [N, 64]
score = query · keyᵀ                                       [N]
logits padded to fixed size (max_neighbors=20 / max_vehicles=50)
→ Categorical distribution → sampled or evaluated action
```

**Action space:** `MultiDiscrete([max_neighbors, max_vehicles])` — one index for the next hop, one for the vehicle.

**Action masking:** Invalid neighbor slots are filled with `-inf` before the softmax so they have zero probability.

---

## 5. PPO Training (`src/models/train.py`)

### Algorithm

The training loop implements PPO with the following components:

| Component | Detail |
|-----------|--------|
| **Rollout collection** | Full episodes stored as `RolloutBuffer` (list of `HeteroData` states) |
| **Advantage estimation** | GAE with γ=0.99, λ=0.92 |
| **Policy update** | Clipped surrogate objective (ε=0.2), 4 update epochs per rollout, mini-batches |
| **Entropy bonus** | Cosine-annealed entropy coefficient to balance exploration vs exploitation |
| **Learning rate** | Cosine-annealed from 3e-4 |
| **Observation normalization** | `RunningMeanStd` reward normalization (from stable-baselines3) |
| **Total episodes** | 5 000 |

### Curriculum Scheduler

Training is split into three phases, each with progressively harder conditions:

| Phase | Label | Anomaly Scale | Max Hops | Vehicles | Advance Condition |
|-------|-------|--------------|----------|----------|-------------------|
| 1 | Easy | 30% | 5 | Trucks only | >70% delivery rate over last 100 eps (min 800 eps in phase) |
| 2 | Medium | 60% | 10 | All | >70% delivery rate (min 1 200 eps in phase) |
| 3 | Full | 100% | 50 | All | — (final phase) |

Forced transition occurs if 2× the minimum episode count is reached regardless of performance.

### Checkpoints

| File | Saved when |
|------|-----------|
| `best_model.pt` | New highest rolling-average reward |
| `latest_model.pt` | Every `checkpoint_interval` episodes |
| `final_model.pt` | End of training |
| `training_metrics.csv` | Every episode |
| `training_curves.png` | End of training |

---

## 6. Simulation Engines

### 6.1 Anomaly Engine (`src/environment/anomaly_engine.py`)

Injects stochastic disruptions at each environment step to prevent the agent from memorizing static paths.

| Type | Default Appear Prob | Disappear Prob | Severity Range | Affects |
|------|-------------------|----------------|----------------|---------|
| Weather | 15% | 10% | 1.2×–2.5× | Edges & Nodes |
| Traffic | 20% | 15% | 1.1×–1.8× | Edges only |
| Sentiment | 8% | 5% | 1.0×–1.5× (+30% cost) | Nodes only |
| Geopolitical | 4% | 3% | 1.5×–3.0× (+100% cost) | Edges & Nodes |

`set_phase(phase)` scales all probabilities by the curriculum scheduler's anomaly scale factor.

### 6.2 Time Engine (`src/environment/time_engine.py`)

Tracks continuous simulation time and modulates travel costs/times:

- **Traffic multipliers** — Rush hours (08:00–10:00, 17:00–20:00) increase travel time up to 1.6×; night travel is faster.
- **Seasonality** — Indian Monsoon (June–September) raises weather anomaly probability.
- **Holidays** — Diwali, Republic Day, Holi, etc. affect demand congestion and fuel prices.

### 6.3 Cost Calculator (`src/environment/cost_calculator.py`)

Computes the full monetary cost (₹) for each leg:

| Component | Formula |
|-----------|---------|
| Fuel | distance × (1/fuel_efficiency) × fuel_price × cargo_weight_factor |
| Toll | route `toll_cost` (fixed per route) |
| Mileage | distance × `mileage_cost_per_km` |
| Maintenance | distance × `maintenance_cost_per_km` × age_factor |
| Mode-switch | Lookup table `MODE_SWITCH_COST[(from_type, to_type)]` (₹0–₹15 000) |
| Insurance | cargo_value × terrain_risk × anomaly_multiplier |

---

## 7. Dashboard (`dashboard/app.py`)

The real-time dashboard is built on **FastAPI** (backend) and **Leaflet.js** (frontend).

### Architecture

```
Browser (Leaflet.js + WebSocket)
        ↕  ws://localhost:8000/ws
FastAPI Server
  └── /ws  — Streams step-by-step simulation events
  └── /     — Serves static HTML/JS/CSS
```

### Features

- **Interactive map** — CartoDB dark-tile base with city nodes and route edges as overlays.
- **Disruption layer** — Highlighted edges/nodes by anomaly type (colour-coded).
- **Optimal path baselines** — Nominal shortest path (dashed green) and dynamic optimal from current position (dashed blue).
- **Live metrics** — Steps, cumulative cost (₹), travel time, and cargo risk updated after every step.
- **Step log** — Full breakdown of each decision: city chosen, vehicle used, leg cost, reward.
- **Agent & scenario selectors** — Switch between Random / Trained GNN+RL and between India / Volatile / Small scenarios.

### WebSocket Event Format

```json
{
  "type": "step",
  "step": 3,
  "current_node": "Nagpur",
  "action": [2, 0],
  "reward": -0.42,
  "done": false,
  "info": {
    "cost": 12340.5,
    "time_hours": 8.2,
    "risk": 0.15
  }
}
```

---

## 8. Configuration Reference

All configuration lives in `src/config/default_config.py`.

### `ScenarioConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | `"default"` | Scenario identifier |
| `locations` | `List[LocationConfig]` | `[]` | City nodes |
| `routes` | `List[RouteConfig]` | `[]` | Transport edges |
| `vehicles` | `List[VehicleConfig]` | `[]` | Fleet |
| `shipment_templates` | `List[ShipmentTemplate]` | `[]` | Random shipment pool |
| `anomaly_config` | `AnomalyConfig` | default | Disruption probabilities & severities |
| `reward_weights` | `RewardWeights` | default | Multi-objective penalty weights |
| `max_steps` | int | `50` | Max hops per episode before truncation |
| `fuel_price` | float | `100.0` | Base fuel price ₹/litre |

### `RewardWeights`

| Field | Default | Meaning |
|-------|---------|---------|
| `time` | 1.0 | Weight on normalised travel-time penalty |
| `cost` | 0.3 | Weight on normalised monetary-cost penalty |
| `risk` | 0.5 | Weight on cargo-risk penalty |
| `spoilage` | 2.0 | Heavy penalty multiplier for shelf-life violation |
| `delay` | 1.5 | Penalty for exceeding expected travel time |

---

## 9. Project Status & Roadmap

### ✅ Completed
- Heterogeneous graph environment with 40 Indian cities and 4 node types.
- Stochastic Anomaly Engine (weather, traffic, sentiment, geopolitical).
- HGT + PPO pipeline with Pointer Network actor heads and GAE.
- 3-phase curriculum learning with automatic phase transitions.
- Real-time WebSocket dashboard with Leaflet.js map.
- Optimal path baselines (static nominal and dynamic from current node).
- Docker + Cloud Run deployment.

### 🚀 Planned
- [ ] **Multi-shipment concurrency** — multiple active shipments competing for vehicle resources.
- [ ] **Warehouse allocation** — agent can park cargo to wait out disruptions.
- [ ] **Real data integration** — OSM road networks, live weather/news APIs.
- [ ] **Explainability layer** — visualise GNN attention weights to show *why* the agent rerouted.
- [ ] **Hyperparameter search** — Optuna integration for automated PPO tuning.
