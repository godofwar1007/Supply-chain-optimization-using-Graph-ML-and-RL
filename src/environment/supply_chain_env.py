"""
Supply Chain Environment — full Gymnasium environment.

The agent routes a single shipment from source to destination across a
graph of Indian cities. At each step, the agent chooses:
  - action[0]: which neighbor to move to (index into neighbors list)
  - action[1]: which vehicle to use (index into available vehicles)

The environment tracks travel time, monetary cost, cargo risk, and
produces observations as flat numpy arrays (gymnasium-compatible) plus
a rich state dict for the GNN feature engine.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from src.config.default_config import (
    ScenarioConfig,
    ShipmentTemplate,
    VehicleConfig,
    TERRAIN_SPEED_FACTOR,
    TERRAIN_RISK_FACTOR,
)
from src.environment.anomaly_engine import AnomalyEngine
from src.environment.time_engine import TimeEngine
from src.environment.cost_calculator import CostCalculator, LegCostBreakdown
from src.utils.graph_utils import (
    build_networkx_graph,
    get_neighbors,
    get_route_config,
    get_all_edge_keys,
)


class SupplyChainEnv(gym.Env):
    """
    Gymnasium environment for supply chain route optimization.

    Observation: flat numpy array encoding shipment, current node, destination,
                 neighbors (edges + nodes), vehicle fleet, and global context.
    Action:      MultiDiscrete([max_neighbors, max_vehicles])
    Reward:      Negative weighted combination of time, cost, and risk.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: ScenarioConfig,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        # Build graph
        self.graph: nx.DiGraph = build_networkx_graph(config)
        self.location_names = config.location_names()
        self.num_locations = len(self.location_names)
        self.loc_to_idx = {name: i for i, name in enumerate(self.location_names)}

        # Compute max degree for observation padding
        self.max_neighbors = max(
            len(list(self.graph.successors(n))) for n in self.graph.nodes
        )
        self.max_vehicles = len(config.vehicles)

        # Sub-engines
        self.anomaly_engine = AnomalyEngine(config.anomaly_config)
        self.time_engine = TimeEngine()
        self.cost_calculator = CostCalculator(config.fuel_price)

        # Vehicle lookup
        self.vehicles: List[VehicleConfig] = config.vehicles
        self.vehicle_positions: Dict[str, str] = {}  # vehicle_id → location

        # ── Observation / action dimensions ────────────────────────────
        # Shipment features: 10
        self.shipment_dim = 10
        # Node features: 10 (9 base + on_nominal_path)
        self.node_dim = 10
        # Edge features: 10
        self.edge_dim = 10
        # Vehicle features: 7
        self.vehicle_dim = 7
        # Global context: 10
        self.context_dim = 10

        self.obs_dim = (
            self.shipment_dim                              # shipment
            + self.node_dim                                # current node
            + self.node_dim                                # destination node
            + self.max_neighbors * (self.edge_dim + self.node_dim)  # neighbor info
            + 1                                            # num valid neighbors
            + self.max_vehicles * self.vehicle_dim         # vehicles
            + 1                                            # num valid vehicles
            + self.context_dim                             # time context
            + 2                                            # progress features
        )

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [self.max_neighbors, self.max_vehicles]
        )

        # ── Episode state ──────────────────────────────────────────────
        self.current_node: str = ""
        self.destination: str = ""
        self.shipment: Optional[ShipmentTemplate] = None
        self.previous_vehicle_type: Optional[str] = None
        self.step_count: int = 0
        self.total_time_hours: float = 0.0
        self.total_cost: float = 0.0
        self.total_risk: float = 0.0
        self.path_taken: List[str] = []
        self.leg_details: List[dict] = []
        self._current_neighbors: List[str] = []
        self._rng = random.Random()

        # Per-node visit counter for escalating loop penalty (×1.5 per revisit)
        self._node_visit_counts: Dict[str, int] = {}

        # ── Curriculum settings ────────────────────────────────────────
        # Controlled by set_curriculum_phase(); defaults = no restriction.
        self._curriculum_max_hops: int = self.config.max_steps
        self._curriculum_vehicle_types: Optional[List[str]] = None  # None = all

        # Pre-compute all-pairs shortest path times for feasibility checks
        # We use base_time_hours as the weight (computed in config.auto_compute_times)
        self._all_pairs_base_times = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='base_time_hours'))

    # ═══════════════════════════════════════════════════════════════════
    # Curriculum control
    # ═══════════════════════════════════════════════════════════════════

    def set_curriculum_phase(
        self,
        phase: int,
        max_hops: int = 50,
        allowed_vehicle_types: Optional[List[str]] = None,
    ) -> None:
        """
        Configure the environment for a curriculum phase.

        Parameters
        ----------
        phase : int
            1 = easy, 2 = medium, 3 = full difficulty.
        max_hops : int
            Maximum number of steps before truncation for this phase.
        allowed_vehicle_types : list[str] or None
            If given, only vehicles of these types are usable by the agent.
            None means all vehicle types are available.
        """
        self._curriculum_max_hops = max_hops
        self._curriculum_vehicle_types = allowed_vehicle_types

    # ═══════════════════════════════════════════════════════════════════
    # Reset
    # ═══════════════════════════════════════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
            np.random.seed(seed)

        # Pick random shipment template first (needed for feasibility check)
        template = self._rng.choice(self.config.shipment_templates)
        self.shipment = ShipmentTemplate(
            product_type=template.product_type,
            fragility=template.fragility,
            shelf_life_hours=template.shelf_life_hours,
            temperature_sensitivity=template.temperature_sensitivity,
            weight_kg=template.weight_kg + self._rng.gauss(0, template.weight_kg * 0.1),
            volume_m3=template.volume_m3,
            priority=template.priority,
            insurance_value=template.insurance_value,
        )

        # Pick random source and destination (ensuring feasibility within shelf life)
        src, dst = self._pick_source_dest(self.shipment)
        self.current_node = src
        self.destination = dst

        # Pre-compute shortest path distances to destination (for reward shaping)
        self._dist_to_dest = {}
        for node in self.location_names:
            try:
                self._dist_to_dest[node] = nx.shortest_path_length(
                    self.graph, node, self.destination
                )
            except nx.NetworkXNoPath:
                self._dist_to_dest[node] = self.config.max_steps

        # Reset vehicle positions to home locations
        self.vehicle_positions = {
            v.vehicle_id: v.home_location for v in self.vehicles
        }

        # Reset engines
        edge_keys = get_all_edge_keys(self.config)
        
        volatile_keys = []
        for r in self.config.routes:
            if r.is_volatile:
                volatile_keys.append((r.source, r.target))
                if r.bidirectional:
                    volatile_keys.append((r.target, r.source))
        
        self.anomaly_engine.initialize(edge_keys, self.location_names, volatile_keys=volatile_keys)
        self.time_engine.randomize(self._rng)

        # Reset episode tracking
        self.previous_vehicle_type = None
        self.previous_node = None  # For anti-backtracking
        self.step_count = 0
        self.total_time_hours = 0.0
        self.total_cost = 0.0
        self.total_risk = 0.0
        self.path_taken = [self.current_node]
        self.leg_details = []

        # Reset per-node visit counter (source counts as 1 visit)
        self._node_visit_counts = {self.current_node: 1}

        # Cache neighbors (no backtrack filter on first step)
        self._current_neighbors = get_neighbors(self.graph, self.current_node)

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def _pick_source_dest(self, shipment: ShipmentTemplate) -> Tuple[str, str]:
        """Pick source and destination ensuring path exists and is feasible."""
        attempts = 0
        while attempts < 100:
            src = self._rng.choice(self.location_names)
            dst = self._rng.choice(self.location_names)
            
            if src == dst:
                continue
                
            # Check if path exists
            if dst not in self._all_pairs_base_times.get(src, {}):
                continue
                
            # Feasibility check: base time should be < 50% of shelf life
            # (leaving 50% for anomalies, mode switches, and sub-optimal routing)
            base_time = self._all_pairs_base_times[src][dst]
            if base_time < shipment.shelf_life_hours * 0.5:
                return src, dst
                
            attempts += 1
            
        # Fallback to a guaranteed short path
        return self.location_names[0], self.location_names[1]

    # ═══════════════════════════════════════════════════════════════════
    # Step
    # ═══════════════════════════════════════════════════════════════════

    def step(self, action):
        next_hop_idx, vehicle_idx = int(action[0]), int(action[1])

        neighbors = self._current_neighbors
        reward = 0.0
        truncated = False

        # ── Validate next-hop ──────────────────────────────────────────
        if next_hop_idx >= len(neighbors):
            # Invalid neighbor → pick the first valid one (penalty)
            next_hop_idx = 0
            reward -= 5.0  # penalty for invalid action

        next_node = neighbors[next_hop_idx]

        # ── Validate vehicle (curriculum-aware) ───────────────────────
        # If curriculum restricts vehicle types, filter the available list.
        if self._curriculum_vehicle_types is not None:
            allowed = [
                v for v in self.vehicles
                if v.vehicle_type in self._curriculum_vehicle_types
            ]
            if not allowed:          # safety fallback
                allowed = self.vehicles
        else:
            allowed = self.vehicles
        vehicle_idx = min(vehicle_idx, len(allowed) - 1)
        vehicle = allowed[vehicle_idx]

        # ── Get route info ─────────────────────────────────────────────
        route = get_route_config(self.config, self.current_node, next_node)

        # ── Compute travel time ────────────────────────────────────────
        terrain_factor = TERRAIN_SPEED_FACTOR.get(route.terrain, 1.0)
        road_factor = route.road_grading
        effective_speed = vehicle.speed_kmph * terrain_factor * road_factor

        # Apply anomaly time multipliers
        anomaly_time = self.anomaly_engine.edge_time_factor(
            self.current_node, next_node
        )
        # Apply time-of-day traffic
        traffic_time = self.time_engine.traffic_factor()
        # Node-level delays (e.g., port congestion)
        node_delay = self.anomaly_engine.node_time_factor(next_node)

        effective_speed = max(effective_speed, 5.0)  # floor to avoid division by zero
        base_travel_hours = route.distance_km / effective_speed
        actual_travel_hours = (
            base_travel_hours * anomaly_time * traffic_time * node_delay
        )

        # Add random noise (log-normal)
        noise = np.random.lognormal(0, 0.05)
        actual_travel_hours *= noise

        # ── Compute cost ───────────────────────────────────────────────
        anomaly_cost = self.anomaly_engine.edge_cost_factor(
            self.current_node, next_node
        )
        edge_risk = self.anomaly_engine.node_risk_score(next_node)
        fuel_factor = self.time_engine.fuel_price_factor()

        cost_breakdown = self.cost_calculator.compute_leg_cost(
            route=route,
            vehicle=vehicle,
            shipment=self.shipment,
            anomaly_cost_factor=anomaly_cost,
            fuel_price_factor=fuel_factor,
            previous_vehicle_type=self.previous_vehicle_type,
            route_risk_score=edge_risk,
        )

        # ── Compute risk ──────────────────────────────────────────────
        terrain_risk = TERRAIN_RISK_FACTOR.get(route.terrain, 0.1)
        leg_risk = (terrain_risk + edge_risk) * self.shipment.fragility

        # ── Update state ──────────────────────────────────────────────
        self.total_time_hours += actual_travel_hours
        self.total_cost += cost_breakdown.total
        self.total_risk += leg_risk
        self.step_count += 1
        self.time_engine.advance(actual_travel_hours)
        self.anomaly_engine.step()
        self.previous_vehicle_type = vehicle.vehicle_type
        self.current_node = next_node
        self.path_taken.append(next_node)

        # Track per-node visit count for escalating loop penalty
        self._node_visit_counts[next_node] = self._node_visit_counts.get(next_node, 0) + 1

        # Log leg details
        self.leg_details.append({
            "from": self.path_taken[-2],
            "to": next_node,
            "vehicle": vehicle.vehicle_id,
            "vehicle_type": vehicle.vehicle_type,
            "time_hours": actual_travel_hours,
            "cost": cost_breakdown,
            "risk": leg_risk,
            "anomalies": self.anomaly_engine.get_edge_anomaly_summary(
                self.path_taken[-2], next_node
            ),
        })

        # ── Check termination ─────────────────────────────────────────
        done = False

        # Reached destination
        if self.current_node == self.destination:
            done = True
            reward += 50.0  # substantial arrival bonus

        # Max steps exceeded (use curriculum max_hops if set)
        effective_max = min(self.config.max_steps, self._curriculum_max_hops)
        if self.step_count >= effective_max:
            truncated = True
            reward -= 15.0  # truncation penalty

        # Shelf life violation
        shelf_life_ratio = self.total_time_hours / self.shipment.shelf_life_hours
        if shelf_life_ratio > 1.0:
            done = True
            reward -= self.config.reward_weights.spoilage * 20.0

        # ── Compute per-step cost penalties ────────────────────────────
        w = self.config.reward_weights

        # Normalize components to comparable scales
        time_penalty = actual_travel_hours / 10.0       # ~10 hrs = 1 unit
        cost_penalty = cost_breakdown.total / 10000.0   # ₹10k = 1 unit
        risk_penalty = leg_risk

        reward -= (
            w.time * time_penalty
            + w.cost * cost_penalty
            + w.risk * risk_penalty
        )

        # Spoilage pressure: increasing penalty as shelf life runs out
        if shelf_life_ratio > 0.7:
            reward -= w.spoilage * (shelf_life_ratio - 0.7) * 2.0

        # ── Potential-based reward shaping (direction toward goal) ─────
        # Uses pre-computed shortest-path hop distance as potential Φ.
        # Reward = Φ(s) - Φ(s') = prev_dist - curr_dist
        # Positive when moving closer, negative when moving away.
        prev_node = self.path_taken[-2]
        prev_dist = self._dist_to_dest.get(prev_node, self.config.max_steps)
        curr_dist = self._dist_to_dest.get(self.current_node, self.config.max_steps)

        direction_reward = (prev_dist - curr_dist) * 5.0
        reward += direction_reward

        # ── Loop penalty & step penalty ───────────────────────────────
        # Escalating penalty: base_penalty × 1.5^(revisits-1)
        # 1st revisit: -5, 2nd: -7.5, 3rd: -11.25, ...
        visit_count = self._node_visit_counts.get(self.current_node, 1)
        if visit_count > 1:
            revisits = visit_count - 1
            reward -= 5.0 * (1.5 ** (revisits - 1)) * revisits

        reward -= 1.0  # Encourage finding the destination faster

        # ── Update neighbors for next step (anti-backtracking) ────────
        self.previous_node = prev_node
        all_neighbors = get_neighbors(self.graph, self.current_node)
        # Remove the node we just came from (unless it's the only neighbor)
        if len(all_neighbors) > 1 and self.previous_node in all_neighbors:
            all_neighbors = [n for n in all_neighbors if n != self.previous_node]
        self._current_neighbors = all_neighbors

        obs = self._build_obs() if not (done or truncated) else np.zeros(
            self.obs_dim, dtype=np.float32
        )
        info = self._build_info()

        return obs, float(reward), done, truncated, info

    # ═══════════════════════════════════════════════════════════════════
    # Observation builder
    # ═══════════════════════════════════════════════════════════════════

    def _build_obs(self) -> np.ndarray:
        """Build flat observation vector."""
        obs_parts = []

        # ── Shipment features (10) ─────────────────────────────────────
        s = self.shipment
        priority_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        remaining_shelf = max(0, 1.0 - self.total_time_hours / s.shelf_life_hours)
        obs_parts.extend([
            hash(s.product_type) % 100 / 100.0,  # product type encoded
            s.fragility,
            min(s.shelf_life_hours / 1000.0, 5.0),  # normalized
            s.temperature_sensitivity,
            s.weight_kg / 10000.0,         # normalized
            s.volume_m3 / 20.0,            # normalized
            s.weight_kg / max(s.volume_m3, 0.01) / 1000.0,  # density norm
            s.insurance_value / 200000.0,  # normalized
            priority_map.get(s.priority, 0.5),
            remaining_shelf,
        ])

        # ── Current node features (8) ──────────────────────────────────
        obs_parts.extend(self._encode_node(self.current_node, is_current=True))

        # ── Destination node features (8) ──────────────────────────────
        obs_parts.extend(self._encode_node(self.destination, is_dest=True))

        # ── Neighbor edges + nodes (max_neighbors × (edge_dim + node_dim)) ─
        neighbors = self._current_neighbors
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                obs_parts.extend(self._encode_edge(self.current_node, neighbors[i]))
                obs_parts.extend(self._encode_node(neighbors[i]))
            else:
                obs_parts.extend([0.0] * (self.edge_dim + self.node_dim))

        # Num valid neighbors (1)
        obs_parts.append(len(neighbors) / self.max_neighbors)

        # ── Vehicle features (max_vehicles × vehicle_dim) ──────────────
        for i in range(self.max_vehicles):
            if i < len(self.vehicles):
                obs_parts.extend(self._encode_vehicle(self.vehicles[i]))
            else:
                obs_parts.extend([0.0] * self.vehicle_dim)

        # Num valid vehicles (1)
        obs_parts.append(len(self.vehicles) / max(self.max_vehicles, 1))

        # ── Global context (10) ────────────────────────────────────────
        obs_parts.extend(self.time_engine.get_context_vector())

        # ── Progress features (2) ─────────────────────────────────────
        obs_parts.append(self.step_count / self.config.max_steps)  # step progress
        obs_parts.append(self.total_time_hours / max(self.shipment.shelf_life_hours, 1))

        obs = np.array(obs_parts, dtype=np.float32)

        # Pad or truncate to exact obs_dim
        if len(obs) < self.obs_dim:
            obs = np.pad(obs, (0, self.obs_dim - len(obs)))
        elif len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]

        return obs

    def _encode_node(
        self, name: str, is_current: bool = False, is_dest: bool = False
    ) -> list:
        """Encode a location node as a feature vector (dim=9)."""
        loc = self.config.location_by_name(name)
        risk = self.anomaly_engine.node_risk_score(name)
        region_map = {"metro": 1.0, "urban": 0.7, "hub": 0.8, "port": 0.6, "rural": 0.3, "coastal": 0.5}
        d = self._dist_to_dest.get(name, self.config.max_steps)
        max_d = max(self._dist_to_dest.values()) if self._dist_to_dest else 10

        return [
            loc.lat / 35.0,          # normalize India lat range ~8-35
            loc.lng / 100.0,         # normalize India lng range ~68-97
            risk,                    # anomaly risk score
            region_map.get(loc.region_type, 0.5),
            float(loc.has_warehouse),
            loc.warehouse_fill_ratio if loc.has_warehouse else 0.0,
            float(is_current),
            float(is_dest),
            d / max(max_d, 1),       # normalized distance to destination
        ]

    def _encode_edge(self, src: str, tgt: str) -> list:
        """Encode a route edge as a feature vector (dim=10)."""
        route = get_route_config(self.config, src, tgt)
        terrain_map = {"flat": 0.2, "hilly": 0.5, "mountainous": 0.8, "coastal": 0.4}
        anomaly_time = self.anomaly_engine.edge_time_factor(src, tgt)
        anomaly_cost = self.anomaly_engine.edge_cost_factor(src, tgt)

        return [
            route.distance_km / 1500.0,        # normalize max ~1500km
            terrain_map.get(route.terrain, 0.5),
            route.road_grading,
            route.toll_cost / 1000.0,           # normalize
            route.mileage_cost_per_km / 10.0,   # normalize
            route.base_time_hours / 20.0,        # normalize
            anomaly_time / 3.0,                  # normalize (max ~3x)
            anomaly_cost / 3.0,
            TERRAIN_RISK_FACTOR.get(route.terrain, 0.1),
            float(tgt == self.destination),       # is this edge going to dest?
        ]

    def _encode_vehicle(self, vehicle: VehicleConfig) -> list:
        """Encode a vehicle as a feature vector (dim=7)."""
        type_map = {"truck": 0.25, "rail": 0.5, "air": 0.75, "ship": 1.0}
        return [
            type_map.get(vehicle.vehicle_type, 0.25),
            vehicle.max_payload_kg / 50000.0,
            vehicle.fuel_efficiency_km_per_l / 10.0,
            vehicle.vehicle_age_years / 15.0,
            vehicle.maintenance_cost_per_km / 5.0,
            vehicle.speed_kmph / 100.0,
            float(self.shipment.weight_kg <= vehicle.max_payload_kg),
        ]

    # ═══════════════════════════════════════════════════════════════════
    # Info / render
    # ═══════════════════════════════════════════════════════════════════

    def _build_info(self) -> dict:
        """Build info dict with rich state for logging/visualization."""
        return {
            "current_node": self.current_node,
            "destination": self.destination,
            "step_count": self.step_count,
            "total_time_hours": self.total_time_hours,
            "total_cost": self.total_cost,
            "total_risk": self.total_risk,
            "path_taken": list(self.path_taken),
            "neighbors": list(self._current_neighbors),
            "num_active_anomalies": self.anomaly_engine.get_all_active_count(),
            "shelf_life_remaining_pct": max(
                0, 100 * (1 - self.total_time_hours / self.shipment.shelf_life_hours)
            ) if self.shipment else 0,
            "shipment_type": self.shipment.product_type if self.shipment else "",
        }

    def get_graph_state(self) -> dict:
        """
        Return the full graph state for the GNN feature engine.

        This is the interface the FeatureEngine uses to build HeteroData.
        """
        # Precompute nominal optimal path using base edge weights only
        # (ignores live anomalies — serves as a stable reference for the GNN).
        try:
            nominal_path = nx.shortest_path(
                self.graph, self.current_node, self.destination,
                weight="base_time_hours",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            nominal_path = []
        nominal_path_set = set(nominal_path)
        nominal_optimal_path = [
            1 if name in nominal_path_set else 0
            for name in self.location_names
        ]

        return {
            "config": self.config,
            "graph": self.graph,
            "current_node": self.current_node,
            "destination": self.destination,
            "shipment": self.shipment,
            "vehicles": self.vehicles,
            "vehicle_positions": dict(self.vehicle_positions),
            "anomaly_engine": self.anomaly_engine,
            "time_engine": self.time_engine,
            "step_count": self.step_count,
            "total_time_hours": self.total_time_hours,
            "dist_to_dest": self._dist_to_dest,
            "neighbors": list(self._current_neighbors),
            "nominal_optimal_path": nominal_optimal_path,
        }

    def render(self):
        """Console rendering for debugging."""
        if self.render_mode != "human":
            return

        if self.step_count == 0:
            print(f"\n{'='*60}")
            print(f"  📦 Shipment: {self.shipment.product_type} "
                  f"({self.shipment.weight_kg:.0f}kg, {self.shipment.priority} priority)")
            print(f"  🚚 Route: {self.path_taken[0]} → {self.destination}")
            print(f"  ⏳ Shelf life: {self.shipment.shelf_life_hours:.0f}h")
            print(f"{'='*60}")
            return

        leg = self.leg_details[-1] if self.leg_details else None
        if leg:
            anom_str = ", ".join(
                f"{a['type']}({a['severity']:.1f}x)" for a in leg["anomalies"]
            ) or "none"
            print(
                f"  Step {self.step_count}: "
                f"{leg['from']} → {leg['to']} "
                f"[{leg['vehicle_type']}] "
                f"| {leg['time_hours']:.1f}h "
                f"| ₹{leg['cost'].total:,.0f} "
                f"| risk {leg['risk']:.2f} "
                f"| anomalies: {anom_str}"
            )

        if self.current_node == self.destination:
            print(f"\n  ✅ DELIVERED in {self.step_count} hops")
            print(f"     Total time:  {self.total_time_hours:.1f}h")
            print(f"     Total cost:  ₹{self.total_cost:,.0f}")
            print(f"     Total risk:  {self.total_risk:.3f}")
            print(f"     Shelf left:  {max(0, self.shipment.shelf_life_hours - self.total_time_hours):.0f}h")
            print(f"     Path: {' → '.join(self.path_taken)}")
