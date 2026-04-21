"""
Configuration dataclasses for the supply chain environment.

All entity types (locations, routes, vehicles, shipments) are defined here
with clean interfaces so real data can be swapped in later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Node configs
# ---------------------------------------------------------------------------

@dataclass
class LocationConfig:
    """A physical point in the supply chain network."""
    name: str
    lat: float
    lng: float
    region_type: str = "urban"          # metro | urban | rural | port | hub
    has_warehouse: bool = False
    warehouse_capacity: float = 0.0     # tonnes (0 if no warehouse)
    warehouse_fill_ratio: float = 0.3   # initial fill (0-1)
    cold_storage: bool = False
    handling_cost: float = 5.0          # ₹/kg
    throughput_rate: float = 100.0      # tonnes/hour


@dataclass
class RouteConfig:
    """A directed edge between two locations."""
    source: str
    target: str
    distance_km: float
    terrain: str = "flat"               # flat | hilly | mountainous | coastal
    road_grading: float = 0.8           # 0-1 (1 = perfect road)
    base_time_hours: float = 0.0        # auto-computed if 0
    toll_cost: float = 0.0              # ₹
    mileage_cost_per_km: float = 5.0    # ₹/km
    bidirectional: bool = True


@dataclass
class VehicleConfig:
    """A transport vehicle in the fleet."""
    vehicle_id: str
    vehicle_type: str = "truck"         # truck | rail | air | ship
    max_payload_kg: float = 10000.0
    fuel_efficiency_km_per_l: float = 5.0
    vehicle_age_years: float = 3.0
    maintenance_cost_per_km: float = 2.0
    speed_kmph: float = 60.0            # base cruising speed
    home_location: str = ""             # where vehicle starts


@dataclass
class ShipmentTemplate:
    """Template for randomly generating shipments."""
    product_type: str = "electronics"
    fragility: float = 0.5              # 0-1
    shelf_life_hours: float = 720.0     # 30 days default
    temperature_sensitivity: float = 0.3 # 0-1
    weight_kg: float = 500.0
    volume_m3: float = 2.0
    priority: str = "medium"            # low | medium | high | critical
    insurance_value: float = 50000.0    # ₹


# ---------------------------------------------------------------------------
# Anomaly / disruption configs
# ---------------------------------------------------------------------------

@dataclass
class AnomalyTypeConfig:
    """Configuration for one anomaly type (weather, traffic, etc.)."""
    prob_appear_per_step: float = 0.1
    prob_disappear_per_step: float = 0.2
    severity_min: float = 1.1           # multiplier on travel time
    severity_max: float = 2.0
    cost_multiplier: float = 1.0        # extra cost factor
    affects: str = "edges"              # edges | nodes | both


@dataclass
class AnomalyConfig:
    """All anomaly types."""
    weather: AnomalyTypeConfig = field(
        default_factory=lambda: AnomalyTypeConfig(
            prob_appear_per_step=0.12,
            prob_disappear_per_step=0.15,
            severity_min=1.2,
            severity_max=2.5,
            affects="both",
        )
    )
    traffic: AnomalyTypeConfig = field(
        default_factory=lambda: AnomalyTypeConfig(
            prob_appear_per_step=0.20,
            prob_disappear_per_step=0.30,
            severity_min=1.1,
            severity_max=1.8,
            affects="edges",
        )
    )
    sentiment: AnomalyTypeConfig = field(
        default_factory=lambda: AnomalyTypeConfig(
            prob_appear_per_step=0.05,
            prob_disappear_per_step=0.08,
            severity_min=1.0,
            severity_max=1.5,
            cost_multiplier=1.3,
            affects="nodes",
        )
    )
    geopolitical: AnomalyTypeConfig = field(
        default_factory=lambda: AnomalyTypeConfig(
            prob_appear_per_step=0.03,
            prob_disappear_per_step=0.05,
            severity_min=1.5,
            severity_max=3.0,
            cost_multiplier=2.0,
            affects="both",
        )
    )


# ---------------------------------------------------------------------------
# Reward / training
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """Weights for the multi-objective reward."""
    time: float = 1.0
    cost: float = 0.3
    risk: float = 0.5
    spoilage: float = 2.0              # heavy penalty for shelf-life violation
    delay: float = 1.5                 # penalty for exceeding expected time


# ---------------------------------------------------------------------------
# Terrain / vehicle lookup tables
# ---------------------------------------------------------------------------

TERRAIN_SPEED_FACTOR: Dict[str, float] = {
    "flat": 1.0,
    "hilly": 0.75,
    "mountainous": 0.55,
    "coastal": 0.85,
}

TERRAIN_RISK_FACTOR: Dict[str, float] = {
    "flat": 0.05,
    "hilly": 0.15,
    "mountainous": 0.25,
    "coastal": 0.10,
}

VEHICLE_TYPE_INDEX: Dict[str, int] = {
    "truck": 0,
    "rail": 1,
    "air": 2,
    "ship": 3,
}

MODE_SWITCH_COST: Dict[tuple, float] = {
    # (from_type, to_type) → ₹ cost for switching
    ("truck", "truck"): 0.0,
    ("truck", "rail"): 5000.0,
    ("truck", "air"): 15000.0,
    ("truck", "ship"): 8000.0,
    ("rail", "truck"): 3000.0,
    ("rail", "rail"): 0.0,
    ("rail", "air"): 12000.0,
    ("rail", "ship"): 6000.0,
    ("air", "truck"): 3000.0,
    ("air", "rail"): 8000.0,
    ("air", "air"): 0.0,
    ("air", "ship"): 10000.0,
    ("ship", "truck"): 4000.0,
    ("ship", "rail"): 6000.0,
    ("ship", "air"): 12000.0,
    ("ship", "ship"): 0.0,
}

# Base fuel price ₹/litre (can be modulated by time engine)
BASE_FUEL_PRICE = 100.0


# ---------------------------------------------------------------------------
# Scenario config (top-level)
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Complete scenario definition — everything the environment needs."""
    name: str = "default"
    locations: List[LocationConfig] = field(default_factory=list)
    routes: List[RouteConfig] = field(default_factory=list)
    vehicles: List[VehicleConfig] = field(default_factory=list)
    shipment_templates: List[ShipmentTemplate] = field(default_factory=list)
    anomaly_config: AnomalyConfig = field(default_factory=AnomalyConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # Environment parameters
    max_steps: int = 50                 # max hops before truncation
    fuel_price: float = BASE_FUEL_PRICE

    def location_names(self) -> List[str]:
        return [loc.name for loc in self.locations]

    def location_by_name(self, name: str) -> LocationConfig:
        for loc in self.locations:
            if loc.name == name:
                return loc
        raise ValueError(f"Location '{name}' not found in scenario")

    def auto_compute_times(self):
        """Fill in base_time_hours from distance and terrain if not set."""
        avg_speed = 60.0  # km/h baseline
        for route in self.routes:
            if route.base_time_hours <= 0:
                terrain_factor = TERRAIN_SPEED_FACTOR.get(route.terrain, 1.0)
                effective_speed = avg_speed * terrain_factor * route.road_grading
                effective_speed = max(effective_speed, 10.0)  # floor
                route.base_time_hours = route.distance_km / effective_speed
