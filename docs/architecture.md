# Architecture Deep-Dive

This document provides a detailed technical description of every component in the GNN + RL pipeline.

---

## 1. System Data Flow

```
Episode start
      │
      ▼
SupplyChainEnv.reset()
  ├── Sample source & destination (feasibility check via Dijkstra)
  ├── Instantiate Shipment from a random ShipmentTemplate
  └── Reset AnomalyEngine, TimeEngine
      │
      ▼ (loop)
env.get_graph_state()  →  raw Python dict
      │
      ▼
FeatureEngine.build(state)  →  HeteroData
      │
      ▼
GNNEncoder.forward(data)
  ├── Type-specific Linear projections  →  hidden_channels
  ├── HGTConv × num_layers              →  hidden_channels
  ├── Output Linear projection          →  out_channels
  └── Graph pooling (locs + shipments) + context_mlp(global_ctx ∥ step_progress)
      → node_embeddings, graph_embedding
      │
      ▼
ActorCritic.forward(data)
  ├── Critic:  graph_embedding → value scalar
  ├── Neighbor head (Pointer Net):
  │     query = MLP([graph_emb ∥ curr_node_emb])
  │     keys  = Linear(neighbor_embeddings)
  │     logits = query · keysᵀ  (padded to max_neighbors=20, masked)
  │     → Categorical(logits)  → a_node
  └── Vehicle head (Pointer Net):
        query = MLP([graph_emb ∥ curr_node_emb])
        keys  = Linear(vehicle_embeddings)
        logits = query · keysᵀ  (padded to max_vehicles=50, masked)
        → Categorical(logits)  → a_veh
      │
      ▼
action = [a_node, a_veh]
      │
      ▼
env.step(action)
  ├── Validate route & vehicle
  ├── Compute travel time (TimeEngine multipliers + log-normal noise)
  ├── Compute cost (CostCalculator)
  ├── Compute risk
  ├── Advance TimeEngine
  ├── Tick AnomalyEngine
  └── Compute reward (step penalty + shaping + spoilage + loop + arrival)
      │
      ▼
RolloutBuffer.append(state, action, log_prob, reward, value, done)
      │  (every N steps or episode end)
      ▼
PPO update
  ├── GAE (γ=0.99, λ=0.92) → advantages, returns
  ├── Mini-batch shuffle
  ├── Clipped surrogate loss (ε=0.2)
  ├── Value loss (MSE)
  └── Entropy bonus (cosine-annealed)
```

---

## 2. Graph Construction

### 2.1 `HeteroData` Schema

```python
data["location"].x          # [num_locations, 10]   location features
data["vehicle"].x           # [num_vehicles,  8]    vehicle features
data["shipment"].x          # [num_shipments, 8]    shipment features

data["location",   "route",           "location"].edge_index   # [2, num_routes]
data["vehicle",    "vehicle_at",      "location"].edge_index   # [2, num_vehicles]
data["shipment",   "shipment_at",     "location"].edge_index   # [2, 1]
data["shipment",   "shipment_dest",   "location"].edge_index   # [2, 1]
# Reverse edges added for bidirectional message passing:
data["location",   "rev_vehicle_at",  "vehicle"].edge_index
data["location",   "rev_shipment_at", "shipment"].edge_index
data["location",   "rev_shipment_dest","shipment"].edge_index

data.global_context   # [1, 10]
data.step_progress    # [1, 2]
data.neighbor_indices # [num_neighbors]  — indices into location nodes
data.current_node_idx # [1]              — index of current location node
```

### 2.2 Location Feature Vector (10 dims)

| Index | Feature | Notes |
|-------|---------|-------|
| 0 | `lat_norm` | Latitude normalised to [0, 1] |
| 1 | `lng_norm` | Longitude normalised to [0, 1] |
| 2 | `region_type` | Encoded: metro=1.0 … rural=0.3 |
| 3 | `warehouse_fill` | 0 if no warehouse |
| 4 | `cold_storage` | Binary |
| 5 | `handling_cost_norm` | Normalised handling cost |
| 6 | `is_current` | 1 if shipment is here |
| 7 | `is_destination` | 1 if this is the target city |
| 8 | `dist_to_dest_norm` | Normalised Dijkstra distance to destination |
| 9 | `on_nominal_path` | 1 if on anomaly-free shortest path |

### 2.3 Vehicle Feature Vector (8 dims)

| Index | Feature |
|-------|---------|
| 0 | `vehicle_type` (truck=0.25 … ship=1.0) |
| 1 | `max_payload_norm` |
| 2 | `fuel_efficiency_norm` |
| 3 | `age_norm` |
| 4 | `maintenance_cost_norm` |
| 5 | `speed_norm` |
| 6 | `is_at_current_location` |
| 7 | `is_active_vehicle` |

### 2.4 Shipment Feature Vector (8 dims)

| Index | Feature |
|-------|---------|
| 0 | `fragility` |
| 1 | `shelf_life_remaining_norm` |
| 2 | `temperature_sensitivity` |
| 3 | `weight_norm` |
| 4 | `volume_norm` |
| 5 | `priority` (encoded: low=0.25 … critical=1.0) |
| 6 | `insurance_value_norm` |
| 7 | `elapsed_time_norm` |

---

## 3. GNN Encoder Detail

### 3.1 HGTConv

PyTorch Geometric's `HGTConv` implements Heterogeneous Graph Transformer convolution:

```
For each edge type (src_type, rel_type, dst_type):
    K, Q, V = type-specific linear projections of src/dst node features
    α = softmax( (Q_dst ∥ K_src) · W_rel / √d )  [multi-head attention]
    msg = V_src · α
Aggregate messages per dst node
Update dst node features with residual connection
```

Two layers of `HGTConv` with 4 attention heads each are used by default (`hidden_channels=64`).

### 3.2 Graph Embedding

The graph-level embedding fuses three signals:

```python
ctx_emb   = context_mlp(cat([global_context, step_progress]))   # [1, 64]
loc_emb   = mean_pool(location_node_embeddings)                  # [1, 64]
ship_emb  = mean_pool(shipment_node_embeddings)                  # [1, 64]
graph_emb = ctx_emb + loc_emb + ship_emb                         # [1, 64]
```

This additive combination allows each signal to contribute independently and is compatible with arbitrary graph sizes.

---

## 4. Actor-Critic Detail

### 4.1 Pointer Network for Neighbor Selection

```python
query_feat = cat([graph_emb, curr_node_emb], dim=-1)   # [1, 128]
q_node     = query_proj_node(query_feat)                # [1, 64]
k_node     = key_proj_node(neighbor_embs)               # [K, 64]
scores     = q_node @ k_node.T                          # [K]
# Pad to max_neighbors=20; set invalid slots to -inf
dist_node  = Categorical(logits=padded_scores)
a_node     = dist_node.sample()
```

The same pattern is used for vehicle selection using `query_proj_veh` / `key_proj_veh` and the full set of vehicle embeddings (padded to `max_vehicles=50`).

### 4.2 Training Objective

```
L = L_policy + 0.5 × L_value - entropy_coef × L_entropy

L_policy  = -min( r(θ)×A, clip(r(θ), 1-ε, 1+ε)×A )   (PPO clip, ε=0.2)
L_value   = MSE(V(s), returns)
L_entropy = mean entropy of both action distributions
```

`entropy_coef` starts at 0.01 and is cosine-annealed to 0.001 over 5 000 episodes.

---

## 5. Curriculum Scheduler

The `CurriculumScheduler` (in `src/models/train.py`) controls both the anomaly engine and environment constraints.

```
Phase 1 (Easy, scale=0.3):
  - All anomaly spawn probs × 0.3
  - max_hops = 5 (prevents the agent from wandering)
  - Only truck vehicles available

Phase 2 (Medium, scale=0.6):
  - All anomaly spawn probs × 0.6
  - max_hops = 10
  - All vehicle types unlocked

Phase 3 (Full, scale=1.0):
  - Full anomaly probabilities
  - max_hops = 50
  - All vehicle types
```

**Transition rule**: Phase advances when the rolling delivery rate over the last 100 episodes exceeds 70%, subject to a minimum episode count per phase. If 2× the minimum count is reached without the 70% threshold, the phase advances anyway.

---

## 6. Scenarios

| Scenario | Cities | Purpose |
|----------|--------|---------|
| `small_scenario()` | 4 | Unit tests, fast CI checks |
| `india_scenario()` | 40 | Full-scale training & demo |
| `volatile_scenario()` | 40 | Stress test — Golden Quadrilateral routes are fast but highly volatile; agent must prefer stable alternatives |
| `reroute_test_scenario()` | 40 | All stochastic anomaly spawn probs set to 0; used for controlled rerouting demos injected manually via the dashboard |

---

## 7. Deployment Architecture

```
Cloud Run (single container, scales to zero)
  └── uvicorn  →  FastAPI app (dashboard/app.py)
       ├── GET  /          → Serves static HTML/JS/CSS
       └── WS   /ws        → Streams simulation events
              │
              └── SupplyChainEnv + ActorCritic (loaded from checkpoints/best_model.pt)
```

The `PORT` environment variable is injected by Cloud Run. The application reads it via `os.environ.get("PORT", 8000)`. The `--no-cpu-throttling` flag is recommended to keep the async simulation loop responsive during active WebSocket sessions.
