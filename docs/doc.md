# Supply Chain Optimization — Full System Design

## Goal

Overhaul the current prototype (fixed-route DQN with MLP) into a production-grade system that:
- Models the supply chain as a **heterogeneous graph** with rich node/edge features
- Uses a **Graph Neural Network (GNN)** to encode the graph state
- Uses a **Reinforcement Learning agent** that consumes GNN embeddings to make routing, vehicle selection, and warehouse allocation decisions
- Incorporates all **12 feature categories** from the design spec

---

## 1. Feature Mapping — Your 12 Categories

Below is each handwritten feature category mapped to where it lives in the system (node feature, edge feature, or global context).

| # | Category | What It Captures | Graph Location |
|---|----------|-----------------|----------------|
| 1 | **Cargo Score** | Product type, fragility, shelf life, temperature sensitivity | **Shipment node** features |
| 2 | **Shipment Material Score** | Weight, volume, density | **Shipment node** features |
| 3 | **Vehicle Type** | Truck, rail, air, ship — categorical | **Vehicle node** features |
| 4 | **Vehicle Charges** | Fuel efficiency, vehicle age, maintenance cost, fuel price | **Vehicle node** features |
| 5 | **Warehouse Capacity** | Current fill level, max capacity, throughput rate | **Warehouse node** features |
| 6 | **Weather Factors** | Temperature, precipitation, wind, visibility, severe alerts | **Edge features** (route-level) + **Node features** (location-level) |
| 7 | **Geographic & Route** | Terrain type, road grading, mileage cost, toll cost | **Edge features** (static) |
| 8 | **Traffic** | Congestion level, time-of-day patterns | **Edge features** (dynamic) |
| 9 | **Sentiment Score (NLP)** | News-derived risk for regions/routes (strikes, unrest, etc.) | **Node features** (location-level risk) |
| 10 | **Time-Based Features** | Seasonality, holiday effects, festival spikes, day-of-week | **Global context** vector |
| 11 | **Cost Mix** | Mode-switch cost, route popularity/demand | **Edge features** + **Global context** |
| 12 | **Insurance & Risk** | Cargo insurance cost, route risk rating, loss probability | **Edge features** + **Shipment node** features |

---

## 2. Graph Structure — Heterogeneous Supply Chain Graph

The supply chain is modeled as a **heterogeneous graph** with multiple node types and edge types.

### 2.1 Node Types

```
┌─────────────────────────────────────────────────────────┐
│                    NODE TYPES                           │
├──────────────┬──────────────────────────────────────────┤
│ Location     │ Cities, ports, hubs — the physical       │
│              │ points in the network                    │
├──────────────┼──────────────────────────────────────────┤
│ Warehouse    │ Storage facilities at locations          │
│              │ (attached to Location nodes)             │
├──────────────┼──────────────────────────────────────────┤
│ Vehicle      │ Available transport vehicles             │
│              │ (each with type, capacity, costs)        │
├──────────────┼──────────────────────────────────────────┤
│ Shipment     │ Active cargo to be routed                │
│              │ (each with cargo score, material score)  │
└──────────────┴──────────────────────────────────────────┘
```

### 2.2 Edge Types

```
┌────────────────────────┬───────────────────────────────────────────┐
│ Edge Type              │ Connects                                  │
├────────────────────────┼───────────────────────────────────────────┤
│ route                  │ Location ↔ Location                       │
│                        │ (terrain, distance, tolls, traffic, etc.) │
├────────────────────────┼───────────────────────────────────────────┤
│ has_warehouse          │ Location → Warehouse                      │
├────────────────────────┼───────────────────────────────────────────┤
│ vehicle_at             │ Vehicle → Location (current position)     │
├────────────────────────┼───────────────────────────────────────────┤
│ shipment_at            │ Shipment → Location (current position)    │
├────────────────────────┼───────────────────────────────────────────┤
│ shipment_dest          │ Shipment → Location (destination)         │
└────────────────────────┴───────────────────────────────────────────┘
```

### 2.3 Feature Vectors Per Node/Edge

#### Location Node Features (dim ≈ 8)
| Feature | Source Category | Type |
|---------|----------------|------|
| latitude, longitude | Geographic (#7) | continuous |
| weather_temperature | Weather (#6) | continuous |
| weather_severity | Weather (#6) | continuous (0-1 scale) |
| sentiment_risk_score | Sentiment/NLP (#9) | continuous (0-1) |
| region_type | Geographic (#7) | categorical (encoded) |
| is_current_position | — | binary |
| is_destination | — | binary |

#### Warehouse Node Features (dim ≈ 5)
| Feature | Source Category | Type |
|---------|----------------|------|
| current_fill_ratio | Warehouse (#5) | continuous (0-1) |
| max_capacity | Warehouse (#5) | continuous |
| throughput_rate | Warehouse (#5) | continuous |
| handling_cost | Cost (#11) | continuous |
| cold_storage_available | Cargo (#1) | binary |

#### Vehicle Node Features (dim ≈ 8)
| Feature | Source Category | Type |
|---------|----------------|------|
| vehicle_type | Vehicle Type (#3) | categorical (one-hot: truck/rail/air/ship) |
| max_payload_kg | Vehicle (#3) | continuous |
| fuel_efficiency | Vehicle Charges (#4) | continuous |
| vehicle_age_years | Vehicle Charges (#4) | continuous |
| maintenance_cost_per_km | Vehicle Charges (#4) | continuous |
| fuel_price_per_unit | Vehicle Charges (#4) | continuous |
| current_load_ratio | — | continuous (0-1) |
| is_available | — | binary |

#### Shipment Node Features (dim ≈ 10)
| Feature | Source Category | Type |
|---------|----------------|------|
| product_type | Cargo Score (#1) | categorical (encoded) |
| fragility_score | Cargo Score (#1) | continuous (0-1) |
| shelf_life_hours | Cargo Score (#1) | continuous |
| temperature_sensitivity | Cargo Score (#1) | continuous |
| weight_kg | Material Score (#2) | continuous |
| volume_m3 | Material Score (#2) | continuous |
| density | Material Score (#2) | continuous |
| insurance_value | Insurance (#12) | continuous |
| priority_level | — | categorical |
| time_remaining_ratio | Time (#10) | continuous (0-1) |

#### Route Edge Features (dim ≈ 12)
| Feature | Source Category | Type |
|---------|----------------|------|
| distance_km | Geographic (#7) | continuous |
| terrain_type | Geographic (#7) | categorical (encoded) |
| road_grading | Geographic (#7) | continuous (0-1) |
| toll_cost | Geographic (#7) | continuous |
| mileage_cost | Geographic (#7) | continuous |
| traffic_congestion | Traffic (#8) | continuous (0-1) |
| weather_impact_factor | Weather (#6) | continuous |
| route_risk_rating | Insurance (#12) | continuous (0-1) |
| route_popularity | Cost (#11) | continuous |
| mode_switch_cost | Cost (#11) | continuous |
| base_travel_time_min | Geographic (#7) | continuous |
| anomaly_factor | Weather/Geopolitical (#6, #9) | continuous |

#### Global Context Vector (dim ≈ 8)
| Feature | Source Category | Type |
|---------|----------------|------|
| day_of_week | Time (#10) | cyclical (sin/cos encoded) |
| month_of_year | Time (#10) | cyclical (sin/cos encoded) |
| is_holiday | Time (#10) | binary |
| is_festival_season | Time (#10) | binary |
| season | Time (#10) | categorical |
| global_fuel_price_index | Vehicle Charges (#4) | continuous |

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                                │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐│
│  │ Feature  │    │  GNN         │    │  RL Agent    │    │Actions ││
│  │ Engine   │───▶│  Encoder     │───▶│  (Policy +   │───▶│        ││
│  │          │    │              │    │   Value)     │    │        ││
│  └──────────┘    └──────────────┘    └──────────────┘    └────────┘│
│       ▲                                     ▲                      │
│       │              ┌──────────┐           │                      │
│       └──────────────│  Env     │───────────┘                      │
│        (obs, graph)  │  (reward)│  (reward, next_state)            │
│                      └──────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Feature Engine (`feature_engine.py`)
- Takes raw config/simulation state and builds the heterogeneous graph
- Normalizes all continuous features
- Encodes categoricals (one-hot or learned embeddings)
- Produces a `torch_geometric.data.HeteroData` object each step

### 3.2 GNN Encoder (`models/gnn_encoder.py`)
- **Architecture**: Heterogeneous Graph Transformer (HGT) or Relational Graph Attention Network (R-GAT)
- Processes each node type with type-specific linear projections
- Runs 2-3 message-passing layers with attention
- Produces per-node embeddings (e.g., 64-dim per node)
- Pools to a graph-level embedding + keeps per-node embeddings for action scoring

```
Input HeteroData
       │
       ▼
┌──────────────────┐
│ Type-specific     │  (project each node type to shared dim)
│ Linear Layers     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ HGT / R-GAT      │  × 2-3 layers
│ Message Passing   │  (edge-type-aware attention)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Node embeddings   │  (per-node 64-dim vectors)
│ + Graph pooling   │  (global graph embedding)
└──────────────────┘
```

### 3.3 RL Agent (`models/rl_agent.py`)

> [!IMPORTANT]
> **Key Design Decision**: The agent's action space needs to be richer than just "pick an edge." We need a **hierarchical or multi-head action** approach.

**Action Space — Three decisions per step:**

| Decision | Description | How It's Scored |
|----------|-------------|-----------------|
| **Next Location** | Which neighboring node to route the shipment to | Score each neighbor node using its GNN embedding |
| **Vehicle Selection** | Which vehicle to use for this leg | Score each available vehicle using its GNN embedding |
| **Warehouse Action** | Store at intermediate warehouse or continue? | Binary head from graph embedding |

**Agent Architecture:**
```
                     ┌─────────────────┐
                     │ Global Context  │
                     │ (time features) │
                     └────────┬────────┘
                              │
Graph Embedding ──────────────┼──────────► Value Head ──► V(s)
                              │
                              ├──────────► Next-Hop Head
                              │            (scores neighbor nodes)
                              │
                              ├──────────► Vehicle Head
                              │            (scores available vehicles)
                              │
                              └──────────► Warehouse Head
                                           (store vs. continue)
```

**RL Algorithm**: PPO (Proximal Policy Optimization)
- More stable than DQN for continuous/complex action spaces
- Handles the multi-head policy naturally
- Better sample efficiency with the GNN encoder

### 3.4 Environment (`environment.py`)

The environment is the core simulation. It needs a major overhaul from the current fixed-route version.

**Key changes from current prototype:**

| Aspect | Current (Prototype) | Full System |
|--------|-------------------|-------------|
| Routing | Fixed route A→B→C→D | Agent chooses next hop from neighbors |
| Vehicle | None | Agent selects vehicle per leg |
| Warehouse | None | Agent can store cargo at intermediate nodes |
| Anomalies | Simple prob. appear/disappear | Rich weather, traffic, sentiment, seasonal |
| Reward | -travel_time only | Multi-objective (time + cost + risk) |
| Observation | Flat vector | HeteroData graph |

**Reward Function (multi-objective):**
```python
reward = -(
    w_time * normalized_travel_time +
    w_cost * normalized_total_cost +       # fuel + tolls + maintenance + mode-switch
    w_risk * normalized_risk_penalty +      # cargo damage prob × insurance value
    w_spoilage * spoilage_penalty +         # shelf-life violation penalty
    w_delay * delivery_delay_penalty        # time-remaining violation
)
```

> [!NOTE]
> The weights (`w_time`, `w_cost`, etc.) can be tuned or even conditioned on shipment priority, allowing the agent to learn different trade-offs.

---

## 4. Module & File Structure

```
supply-chain-optimization/
│
├── pyproject.toml
├── README.md
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── default_config.py        # Default graph topology, feature ranges
│   │   └── scenarios.py             # Pre-built scenarios (small, medium, India-map, etc.)
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── supply_chain_env.py      # Main Gymnasium env (HeteroData obs)
│   │   ├── anomaly_engine.py        # Weather, traffic, sentiment anomaly simulation
│   │   ├── cost_calculator.py       # Fuel, toll, maintenance, insurance cost logic
│   │   └── time_engine.py           # Seasonality, holidays, time-of-day effects
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_engine.py        # Builds HeteroData from env state
│   │   ├── normalizers.py           # Feature normalization utilities
│   │   └── encoders.py              # Categorical encoding (terrain, vehicle type, etc.)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gnn_encoder.py           # HGT / R-GAT encoder
│   │   ├── rl_agent.py              # PPO agent with multi-head actions
│   │   ├── action_heads.py          # Next-hop, vehicle, warehouse action heads
│   │   └── baseline.py              # Greedy / shortest-path baselines for comparison
│   │
│   └── utils/
│       ├── __init__.py
│       ├── graph_utils.py           # NetworkX ↔ PyG conversion helpers
│       ├── visualization.py         # Graph plotting, training curves, route viz
│       └── metrics.py               # Evaluation metrics (avg time, cost, risk, etc.)
│
├── train.py                         # Training entry point (PPO loop)
├── evaluate.py                      # Evaluation & benchmarking
├── tune.py                          # Optuna hyperparameter search
│
├── tests/
│   ├── test_environment.py
│   ├── test_feature_engine.py
│   └── test_agent.py
│
└── notebooks/                       # Jupyter exploration
    └── exploration.ipynb
```

---

## 5. Phased Implementation Roadmap

### Phase 1 — Graph Environment & Feature Engine
Build the new environment and feature pipeline. No ML yet — just the simulation.

- [ ] Design the config schema for a full supply chain graph (nodes, edges, all 12 feature categories)
- [ ] Build `supply_chain_env.py` with dynamic routing (agent picks next hop)
- [ ] Build `anomaly_engine.py` (weather, traffic, sentiment, geopolitical — all stochastic)
- [ ] Build `time_engine.py` (seasonality, holidays, day-of-week effects)
- [ ] Build `cost_calculator.py` (fuel, tolls, maintenance, insurance, mode-switch)
- [ ] Build `feature_engine.py` (env state → `HeteroData`)
- [ ] Create test scenarios (small 4-node, medium 10-node)
- [ ] Verify environment with random agent

### Phase 2 — GNN Encoder
Build and test the graph encoder in isolation.

- [ ] Implement type-specific input projections
- [ ] Implement HGT or R-GAT message passing layers
- [ ] Implement graph pooling (mean pool + attention pool)
- [ ] Unit test: verify forward pass with sample `HeteroData`

### Phase 3 — RL Agent (PPO + GNN)
Wire up the full agent.

- [ ] Implement multi-head policy (next-hop, vehicle, warehouse)
- [ ] Implement PPO with GAE (Generalized Advantage Estimation)
- [ ] Implement action masking (mask invalid next-hops, unavailable vehicles)
- [ ] Train on small scenario, verify learning signal

### Phase 4 — Training & Tuning
Scale up training and optimize.

- [ ] Train on medium/large scenarios
- [ ] Optuna sweep for GNN + PPO hyperparameters
- [ ] Compare against baselines (greedy shortest-path, random)
- [ ] Ablation study: which feature categories matter most?

### Phase 5 — Visualization & Polish
- [ ] Route visualization on graph (networkx + matplotlib)
- [ ] Training dashboard (reward, cost, risk curves)
- [ ] Write comprehensive README

---

## Open Questions

> [!IMPORTANT]
> **Q1: Simulation vs. Real Data?**  
> The current system is fully simulated. Do you plan to eventually feed in real-world data (e.g., actual road network data from OpenStreetMap, real weather APIs, news sentiment APIs)? This affects how we design the data ingestion layer. For now, I'll design a clean simulation that *mirrors* real data shapes so it's easy to swap in later.

> [!IMPORTANT]
> **Q2: Scale — How large should the graph be?**  
> The prototype has 4 nodes. Are you targeting:
> - **Small** (4-10 nodes) — proof of concept
> - **Medium** (20-50 nodes) — realistic regional network
> - **Large** (100+ nodes) — national/international scale
>
> This affects GNN architecture choices (attention vs. simpler convolutions) and training compute.

> [!WARNING]
> **Q3: NLP Sentiment (Feature #9)**  
> Implementing real-time news sentiment scoring is a significant sub-project on its own. For the initial build, I'd recommend **simulating** the sentiment score as a stochastic signal (similar to how weather anomalies work now), with the interface designed so a real NLP pipeline can be plugged in later. Sound good?

> [!IMPORTANT]
> **Q4: Multi-shipment or single-shipment per episode?**  
> Should the agent route one shipment at a time, or manage multiple concurrent shipments competing for vehicles and warehouse space? Multi-shipment is more realistic but significantly more complex.

---

## Verification Plan

### Automated Tests
- Unit tests for each module (`pytest tests/`)
- Environment smoke test: run 100 episodes with random agent, verify no crashes and reward is bounded
- Feature engine test: verify `HeteroData` shapes match expected dimensions
- GNN forward pass test: verify output dimensions
- Training convergence test: agent should beat random baseline within 200 episodes on small scenario

### Manual Verification
- Visualize trained agent's routing decisions on the graph
- Compare total cost/time/risk against greedy shortest-path baseline
- Inspect GNN attention weights to verify the model is using relevant features
