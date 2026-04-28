# API Reference

Module-level reference for the `src/` package.

---

## `src.config.default_config`

Configuration dataclasses used throughout the project. Import with:

```python
from src.config.default_config import (
    LocationConfig, RouteConfig, VehicleConfig,
    ShipmentTemplate, AnomalyConfig, RewardWeights, ScenarioConfig,
)
```

### `LocationConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Unique city identifier |
| `lat` | `float` | — | Latitude |
| `lng` | `float` | — | Longitude |
| `region_type` | `str` | `"urban"` | One of `metro`, `urban`, `hub`, `port`, `rural`, `coastal` |
| `has_warehouse` | `bool` | `False` | Whether this city has a warehouse |
| `warehouse_capacity` | `float` | `0.0` | Warehouse capacity in tonnes |
| `warehouse_fill_ratio` | `float` | `0.3` | Initial fill ratio (0–1) |
| `cold_storage` | `bool` | `False` | Cold chain available |
| `handling_cost` | `float` | `5.0` | ₹/kg for loading/unloading |
| `throughput_rate` | `float` | `100.0` | tonnes/hour processing capacity |

### `RouteConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | `str` | — | Source city name |
| `target` | `str` | — | Target city name |
| `distance_km` | `float` | — | Road distance in km |
| `terrain` | `str` | `"flat"` | One of `flat`, `hilly`, `mountainous`, `coastal` |
| `road_grading` | `float` | `0.8` | Road quality (0–1, 1 = perfect) |
| `base_time_hours` | `float` | `0.0` | Auto-computed if 0 via `ScenarioConfig.auto_compute_times()` |
| `toll_cost` | `float` | `0.0` | Fixed toll in ₹ |
| `mileage_cost_per_km` | `float` | `5.0` | Variable cost in ₹/km |
| `bidirectional` | `bool` | `True` | If True, adds reverse edge automatically |
| `is_volatile` | `bool` | `False` | Marks route as prone to high-severity anomalies |

### `VehicleConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vehicle_id` | `str` | — | Unique vehicle identifier |
| `vehicle_type` | `str` | `"truck"` | One of `truck`, `rail`, `air`, `ship` |
| `max_payload_kg` | `float` | `10000.0` | Maximum cargo weight |
| `fuel_efficiency_km_per_l` | `float` | `5.0` | Fuel economy in km/L |
| `vehicle_age_years` | `float` | `3.0` | Age affects maintenance cost |
| `maintenance_cost_per_km` | `float` | `2.0` | ₹/km |
| `speed_kmph` | `float` | `60.0` | Base cruising speed |
| `home_location` | `str` | `""` | Starting city name |

### `ShipmentTemplate`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `product_type` | `str` | `"electronics"` | Cargo category identifier |
| `fragility` | `float` | `0.5` | 0–1 (1 = extremely fragile) |
| `shelf_life_hours` | `float` | `720.0` | Max delivery window in hours |
| `temperature_sensitivity` | `float` | `0.3` | 0–1 |
| `weight_kg` | `float` | `500.0` | Cargo weight |
| `volume_m3` | `float` | `2.0` | Cargo volume |
| `priority` | `str` | `"medium"` | One of `low`, `medium`, `high`, `critical` |
| `insurance_value` | `float` | `50000.0` | ₹ declared value |

### `AnomalyTypeConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prob_appear_per_step` | `float` | `0.1` | Probability of spawning at each step |
| `prob_disappear_per_step` | `float` | `0.2` | Probability of expiring at each step |
| `severity_min` | `float` | `1.1` | Minimum travel-time multiplier |
| `severity_max` | `float` | `2.0` | Maximum travel-time multiplier |
| `cost_multiplier` | `float` | `1.0` | Extra cost factor when active |
| `affects` | `str` | `"edges"` | One of `edges`, `nodes`, `both` |

### `RewardWeights`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `time` | `float` | `1.0` | Weight on travel-time penalty |
| `cost` | `float` | `0.3` | Weight on monetary-cost penalty |
| `risk` | `float` | `0.5` | Weight on cargo-risk penalty |
| `spoilage` | `float` | `2.0` | Multiplier on shelf-life violation penalty |
| `delay` | `float` | `1.5` | Penalty for exceeding expected travel time |

### `ScenarioConfig`

Top-level container that assembles all of the above.

```python
cfg = ScenarioConfig(
    name="my_scenario",
    locations=[...],
    routes=[...],
    vehicles=[...],
    shipment_templates=[...],
    max_steps=50,
)
cfg.auto_compute_times()  # Fill in base_time_hours from distance + terrain
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `location_names()` | `List[str]` | Names of all cities |
| `location_by_name(name)` | `LocationConfig` | Lookup by name (raises ValueError if missing) |
| `auto_compute_times()` | `None` | Fills `base_time_hours` for routes where it is 0 |

---

## `src.config.scenarios`

Pre-built scenario factories.

```python
from src.config.scenarios import (
    small_scenario,
    india_scenario,
    volatile_scenario,
    reroute_test_scenario,
)
```

| Function | Nodes | Description |
|----------|-------|-------------|
| `small_scenario()` | 4 | Minimal graph for unit tests |
| `india_scenario()` | 40 | Full Indian logistics network |
| `volatile_scenario()` | 40 | India with highly volatile Golden Quadrilateral routes |
| `reroute_test_scenario()` | 40 | India with all stochastic anomalies disabled |

---

## `src.environment.supply_chain_env`

### `SupplyChainEnv(config, render_mode=None)`

Custom Gymnasium environment.

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `ScenarioConfig` | Scenario definition |
| `render_mode` | `str \| None` | `"human"` for console logging, `None` for silent |

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `observation_space` | `Box` | Flat observation array |
| `action_space` | `MultiDiscrete([max_neighbors, max_vehicles])` | Next hop + vehicle selection |
| `max_neighbors` | `int` | Maximum neighbour index (used to size action space) |
| `max_vehicles` | `int` | Maximum vehicle index |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `reset(seed=None, options=None)` | `(obs, info)` | Start a new episode |
| `step(action)` | `(obs, reward, done, truncated, info)` | Execute one routing decision |
| `render()` | `None` | Print current step summary to stdout |
| `get_graph_state()` | `dict` | Full environment state as a Python dict for `FeatureEngine` |
| `set_curriculum_phase(phase, max_hops, vehicle_types)` | `None` | Apply curriculum constraints |

---

## `src.environment.anomaly_engine`

### `AnomalyEngine(config)`

Manages stochastic disruptions.

| Method | Description |
|--------|-------------|
| `tick()` | Advance one step — spawn and expire anomalies |
| `get_edge_modifier(src, tgt)` | `float` — combined severity multiplier for an edge |
| `get_node_modifier(node)` | `float` — combined severity multiplier for a node |
| `set_phase(phase)` | Scale all spawn probabilities by the curriculum phase factor |
| `reset()` | Clear all active anomalies |

---

## `src.environment.time_engine`

### `TimeEngine()`

Tracks simulation time and computes multipliers.

| Method | Returns | Description |
|--------|---------|-------------|
| `advance(hours)` | `None` | Advance simulation clock |
| `get_traffic_multiplier()` | `float` | Time-of-day travel-time multiplier |
| `get_fuel_price_multiplier()` | `float` | Seasonal fuel price factor |
| `is_monsoon()` | `bool` | True during June–September |
| `is_holiday()` | `bool` | True on Indian public holidays |
| `reset()` | `None` | Reset clock to simulation start |

---

## `src.environment.cost_calculator`

### `CostCalculator(config)`

Computes per-leg monetary costs.

| Method | Returns | Description |
|--------|---------|-------------|
| `compute(route, vehicle, shipment, anomaly_multiplier, time_engine)` | `dict` | Full cost breakdown: fuel, toll, mileage, maintenance, mode_switch, insurance, total |

---

## `src.features.feature_engine`

### `FeatureEngine()`

Converts environment state dicts into `HeteroData` objects.

| Method | Returns | Description |
|--------|---------|-------------|
| `build(state_dict)` | `HeteroData` | Build graph from `env.get_graph_state()` output |

---

## `src.models.gnn_encoder`

### `GNNEncoder(metadata, hidden_channels=64, out_channels=64, num_heads=4, num_layers=2)`

HGT-based graph encoder.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | `tuple` | — | `(node_types, edge_types)` from `data.metadata()` |
| `hidden_channels` | `int` | `64` | Intermediate embedding dimension |
| `out_channels` | `int` | `64` | Output embedding dimension |
| `num_heads` | `int` | `4` | Attention heads per HGTConv layer |
| `num_layers` | `int` | `2` | Number of message-passing layers |

**`forward(data) → dict`**

Returns:
```python
{
    "node_embeddings": {
        "location":  Tensor[num_locations, out_channels],
        "vehicle":   Tensor[num_vehicles,  out_channels],
        "shipment":  Tensor[num_shipments, out_channels],
    },
    "graph_embedding": Tensor[1, out_channels],
}
```

---

## `src.models.ppo_agent`

### `ActorCritic(metadata, hidden_channels=64, out_channels=64, num_heads=4, num_layers=2, max_neighbors=20, max_vehicles=50)`

Actor-Critic with Pointer Network action heads.

**`forward(data, action=None)`**

- If `action` is `None`: samples a new action.
  - Returns `(action, log_prob, entropy, value)`
- If `action` is provided: evaluates its log-probability.
  - Returns `(log_prob, entropy, value)`

| Output | Shape | Description |
|--------|-------|-------------|
| `action` | `[2]` | `[neighbor_idx, vehicle_idx]` |
| `log_prob` | `[1]` | Sum of log-probs of both action dimensions |
| `entropy` | `[1]` | Sum of entropies of both distributions |
| `value` | `[1]` | Estimated state value |

---

## `src.utils.graph_utils`

Utility functions for graph operations.

| Function | Returns | Description |
|----------|---------|-------------|
| `shortest_path_distance(graph, src, dst)` | `float` | Dijkstra distance on nominal edge weights |
| `get_neighbors(graph, node)` | `List[str]` | Adjacent city names |

---

## CLI (`main.py`)

```
python main.py <command>

Commands:
  train      Train PPO agent with 3-phase curriculum (5 000 episodes)
  eval       Run one episode with the best saved checkpoint
  random     Run one episode with random actions (baseline)
  dashboard  Start FastAPI server on port 8000 (or $PORT)
```
