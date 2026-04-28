# Supply Chain Simulation Engine

This document details the mechanics of the custom Gymnasium environment (`SupplyChainEnv`) used for supply chain routing optimization using Reinforcement Learning (RL). The simulation models a realistic logistics network across Indian cities, integrating spatial, temporal, economic, and stochastic elements.

## 1. High-Level Overview

The agent's objective is to successfully route a shipment (cargo) from a randomly selected source city to a destination city across a transportation graph. At each step, the agent must make two simultaneous decisions:
1. **Which neighboring city to travel to next.**
2. **Which vehicle mode to use for that leg.**

The environment is designed as an episodic Markov Decision Process (MDP). If the shipment takes too long and exceeds its "shelf life," or if the agent takes too many steps (`max_steps=50` for the India scenario), the episode is truncated. A successful episode results in the shipment arriving at its destination while minimizing time, financial cost, and cargo risk.

---

## 2. Spatial Network (The Graph)

The world is modeled as a directed graph where:

- **Nodes** represent cities or key logistics hubs. Nodes carry metadata such as geographical coordinates (lat/lng), region type (metro, port, rural, etc.), and warehouse utilization (if present).
- **Edges** represent transportation routes between cities. Edges have physical characteristics: distance (km), terrain type (flat, hilly, mountainous, coastal), road grading (quality 0–1), toll costs, and base travel time.

At the beginning of an episode, the environment randomly samples a source and destination. It performs a feasibility check — using Dijkstra on nominal edge weights — to ensure the destination is reachable within 70% of the shipment's shelf life under base conditions. If the check fails, a new pair is sampled.

---

## 3. Sub-Engines

The environment relies on three underlying engines to calculate the complex realities of the supply chain at every step.

### 3.1 Time Engine (`src/environment/time_engine.py`)

The simulation tracks continuous time (hours, days, months) and uses it to modulate travel efficiency and costs.

- **Traffic patterns**: Applies time-of-day multipliers. Travel during rush hours (08:00–10:00, 17:00–20:00) significantly increases travel time (up to 1.6×), while night travel is faster.
- **Seasonality & weather**: Checks for the Indian Monsoon season (June–September), which raises the probability of weather-related anomalies.
- **Economic cycles**: Factors in Indian public holidays (Diwali, Republic Day, Holi, etc.) and festival seasons, which affect demand congestion and fuel prices.

### 3.2 Anomaly Engine (`src/environment/anomaly_engine.py`)

To prevent the RL agent from simply memorizing static shortest paths, the `AnomalyEngine` injects stochastic disruptions at every step.

- **Disruption types**: Weather, traffic jams, geopolitical events, and social sentiment shifts.
- **Mechanics**: Anomalies spawn and expire probabilistically on edges (routes) and/or nodes (cities). Each active anomaly carries a severity scalar drawn uniformly from `[severity_min, severity_max]`.
- **Impact**: Active anomalies multiply the base travel time and cost of affected edges/nodes by the severity scalar.
- **Curriculum scaling**: `set_phase(phase)` multiplies all spawn probabilities by a phase-specific scale factor (0.3 / 0.6 / 1.0), so the agent faces fewer disruptions early in training and progressively harder conditions later.

Default probability parameters (Phase 3 / Full):

| Anomaly | Appear/step | Disappear/step | Severity Range | Affects |
|---------|-------------|----------------|----------------|---------|
| Weather | 15% | 10% | 1.2× – 2.5× | Edges & Nodes |
| Traffic | 20% | 15% | 1.1× – 1.8× | Edges only |
| Sentiment | 8% | 5% | 1.0× – 1.5× | Nodes only |
| Geopolitical | 4% | 3% | 1.5× – 3.0× | Edges & Nodes |

### 3.3 Cost Calculator (`src/environment/cost_calculator.py`)

Computes the exact monetary cost (₹) for every leg traveled. It accounts for:

- **Fuel cost**: `distance × (1/fuel_efficiency) × fuel_price × weight_factor`.
- **Toll & mileage**: Fixed toll per route + per-km variable cost.
- **Maintenance**: `distance × maintenance_cost_per_km × age_factor`.
- **Mode-switch penalty**: Transferring cargo between vehicle types (e.g., truck → rail) incurs a one-time transshipment cost of ₹3 000–₹15 000 depending on the mode pair.
- **Insurance**: Dynamically calculated from the cargo's declared value, the terrain's inherent risk factor, and any active anomaly multipliers.

---

## 4. Agent Interaction

### 4.1 Observation Space

At each step the environment exposes a flat NumPy array (for tabular / MLP baselines) and also a `HeteroData` graph object (for the GNN agent) via `env.get_graph_state()`. The graph state includes:

- **Location nodes**: Normalised lat/lng, region type encoding, warehouse fill ratio, cold storage flag, handling cost, `on_nominal_path` flag.
- **Vehicle nodes**: Type encoding, payload, speed, fuel efficiency, maintenance cost.
- **Shipment node**: Fragility, remaining shelf life (normalised), weight, priority, temperature sensitivity.
- **Edge features** (encoded into `route` edge attributes): Distance, terrain, road grading, toll cost, active anomaly multipliers.
- **Global context** tensor `[1, 10]`: Cyclical time encodings (sin/cos of hour, day, month), monsoon flag, holiday flag, fuel price multiplier.
- **Step progress** tensor `[1, 2]`: Step fraction and remaining shelf-life fraction.

### 4.2 Action Space

`gymnasium.spaces.MultiDiscrete([max_neighbors, max_vehicles])`.

- `action[0]` — index into the sorted neighbor list (valid range 0..num_neighbors-1).
- `action[1]` — index into the vehicle list at the current location.

If the agent selects an out-of-range neighbor index, a severe penalty (−10) is applied and the agent is forced to take the first valid neighbor instead.

---

## 5. Step Logic & Reward Mechanism

When the agent submits an action, `step()` executes the following sequence:

1. **Validation & configuration** — Validates the chosen route and vehicle. Pulls the physical route configuration from the scenario.
2. **Speed & time calculation**:
   - Effective speed: `base_speed × terrain_factor × road_grading`.
   - Actual time: Applies multipliers from traffic patterns, active weather/traffic anomalies, and log-normal noise for realism.
3. **Cost & risk calculation** — Invokes `CostCalculator` for the monetary breakdown. Computes cargo risk from terrain, active anomalies, and shipment fragility.
4. **State update** — Advances `TimeEngine` by the leg's travel hours. Ticks `AnomalyEngine` to spawn/expire disruptions. Updates the shipment's current location.
5. **Reward calculation** — Composed of the following terms:

| Term | Sign | Description |
|------|------|-------------|
| Step penalty | − | Weighted sum of normalized time, cost, and risk: `−(w_time×t + w_cost×c + w_risk×r)` |
| Spoilage penalty | − | Escalating penalty as remaining shelf life falls below threshold; massive if shelf life = 0 |
| Potential-based shaping | ± | `+5.0 × (distance_before − distance_after)` to the destination |
| Loop penalty | − | Escalating penalty for revisiting the same node |
| Invalid action penalty | − | −10 if agent chose an invalid neighbor index |
| Arrival bonus | + | +50 one-time bonus on reaching the destination |

**Reward weights** (configurable via `RewardWeights`):

| Weight | Default | Effect |
|--------|---------|--------|
| `time` | 1.0 | Penalises long legs |
| `cost` | 0.3 | Penalises expensive legs |
| `risk` | 0.5 | Penalises risky terrain/anomalies |
| `spoilage` | 2.0 | Heavy penalty for shelf-life violations |
| `delay` | 1.5 | Penalty for exceeding expected travel time |

By balancing these diverse constraints the RL agent learns to navigate dynamic supply chain shocks: avoiding bad weather, routing around traffic, and switching to cheaper or faster vehicle modes only when economically viable.

---

## 6. Curriculum Learning

Training uses a `CurriculumScheduler` that automatically progresses through three phases:

| Phase | Anomaly Scale | Max Hops | Vehicle Types | Advance Condition |
|-------|--------------|----------|---------------|-------------------|
| 1 — Easy | 30% | 5 | Trucks only | >70% delivery rate (min 800 eps in phase) |
| 2 — Medium | 60% | 10 | All | >70% delivery rate (min 1 200 eps in phase) |
| 3 — Full | 100% | 50 | All | Final phase |

If 2× the minimum episode count for a phase is exceeded without reaching the 70% threshold, the scheduler forces the transition anyway to prevent the agent from being stuck in a phase indefinitely.

---

## 7. Episode Termination Conditions

| Condition | Outcome |
|-----------|---------|
| Agent reaches destination | `done=True`, +50 arrival bonus |
| Shelf life exhausted (remaining ≤ 0) | `done=True`, large spoilage penalty |
| Steps exceed `max_steps` | `truncated=True`, no bonus |
