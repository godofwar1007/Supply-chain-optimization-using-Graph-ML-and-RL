"""
Pre-built scenarios for the supply chain environment.

- small_scenario():  4 nodes for unit tests / fast iteration
- india_scenario():  25 Indian cities — medium-scale demo scenario
"""

from __future__ import annotations

import math
from typing import List

from .default_config import (
    LocationConfig,
    RouteConfig,
    VehicleConfig,
    ShipmentTemplate,
    ScenarioConfig,
    AnomalyConfig,
    RewardWeights,
)


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _make_route(
    src: str, tgt: str,
    locations: dict,
    terrain: str = "flat",
    road_grading: float = 0.8,
    toll_cost: float = 200.0,
    mileage_cost_per_km: float = 5.0,
    bidirectional: bool = True,
) -> RouteConfig:
    """Build a RouteConfig, auto-computing distance from lat/lng."""
    s = locations[src]
    t = locations[tgt]
    dist = _haversine(s.lat, s.lng, t.lat, t.lng)
    # Road distance is ~1.3x straight-line
    road_dist = dist * 1.3
    return RouteConfig(
        source=src,
        target=tgt,
        distance_km=round(road_dist, 1),
        terrain=terrain,
        road_grading=road_grading,
        toll_cost=toll_cost,
        mileage_cost_per_km=mileage_cost_per_km,
        bidirectional=bidirectional,
    )


# ═══════════════════════════════════════════════════════════════════════
# Small test scenario (4 nodes)
# ═══════════════════════════════════════════════════════════════════════

def small_scenario() -> ScenarioConfig:
    """Minimal 4-node scenario for unit tests."""
    locations = [
        LocationConfig("A", 28.61, 77.21, "metro", True, 5000, 0.4, True, 8.0, 200),
        LocationConfig("B", 26.91, 75.79, "urban", True, 2000, 0.5, False, 5.0, 100),
        LocationConfig("C", 23.02, 72.57, "metro", True, 4000, 0.3, True, 7.0, 180),
        LocationConfig("D", 19.08, 72.88, "metro", True, 6000, 0.2, True, 10.0, 250),
    ]
    routes = [
        RouteConfig("A", "B", 270, "flat", 0.85, 150, 5.0),
        RouteConfig("A", "C", 660, "hilly", 0.70, 400, 5.5),
        RouteConfig("B", "C", 400, "flat", 0.80, 250, 5.0),
        RouteConfig("B", "D", 700, "mountainous", 0.60, 500, 6.0),
        RouteConfig("C", "D", 530, "coastal", 0.75, 300, 5.0),
    ]
    vehicles = [
        VehicleConfig("truck_1", "truck", 10000, 5.0, 2, 2.0, 60, "A"),
        VehicleConfig("truck_2", "truck", 8000, 6.0, 5, 3.0, 55, "B"),
    ]
    templates = [
        ShipmentTemplate("electronics", 0.7, 720, 0.3, 500, 2.0, "medium", 50000),
    ]
    cfg = ScenarioConfig(
        name="small_test",
        locations=locations,
        routes=routes,
        vehicles=vehicles,
        shipment_templates=templates,
        max_steps=20,
    )
    cfg.auto_compute_times()
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# India scenario (25 nodes)
# ═══════════════════════════════════════════════════════════════════════

def india_scenario() -> ScenarioConfig:
    """
    Medium-scale Indian supply chain network.

    25 cities with realistic lat/lng, warehouse placements at major hubs,
    and a fleet of trucks + rail vehicles distributed across the network.
    Designed to look great on a map-based dashboard.
    """

    locations = [
        # Major metros — full warehouses with cold storage
        LocationConfig("Delhi",       28.6139, 77.2090, "metro", True, 8000, 0.35, True, 8.0, 300),
        LocationConfig("Mumbai",      19.0760, 72.8777, "metro", True, 10000, 0.40, True, 10.0, 350),
        LocationConfig("Chennai",     13.0827, 80.2707, "metro", True, 6000, 0.30, True, 8.0, 250),
        LocationConfig("Kolkata",     22.5726, 88.3639, "metro", True, 5000, 0.45, True, 7.0, 220),
        LocationConfig("Bangalore",   12.9716, 77.5946, "metro", True, 7000, 0.25, True, 9.0, 280),
        LocationConfig("Hyderabad",   17.3850, 78.4867, "metro", True, 6000, 0.30, True, 8.0, 260),

        # Secondary cities — warehouses, some cold storage
        LocationConfig("Pune",        18.5204, 73.8567, "urban", True, 3000, 0.35, False, 6.0, 150),
        LocationConfig("Ahmedabad",   23.0225, 72.5714, "urban", True, 4000, 0.40, True, 6.0, 180),
        LocationConfig("Jaipur",      26.9124, 75.7873, "urban", True, 2500, 0.30, False, 5.0, 120),
        LocationConfig("Lucknow",     26.8467, 80.9462, "urban", True, 3000, 0.35, False, 5.5, 140),
        LocationConfig("Chandigarh",  30.7333, 76.7794, "urban", True, 2000, 0.25, False, 5.0, 100),
        LocationConfig("Nagpur",      21.1458, 79.0882, "hub",   True, 5000, 0.30, True, 6.0, 200),
        LocationConfig("Indore",      22.7196, 75.8577, "urban", True, 2000, 0.35, False, 5.0, 110),

        # Smaller / strategic nodes — some with warehouses
        LocationConfig("Bhopal",         23.2599, 77.4126, "urban", True, 1500, 0.30, False, 4.5, 90),
        LocationConfig("Visakhapatnam",  17.6868, 83.2185, "port",  True, 3000, 0.25, True, 6.0, 160),
        LocationConfig("Kochi",          9.9312,  76.2673, "port",  True, 3500, 0.20, True, 7.0, 170),
        LocationConfig("Guwahati",       26.1445, 91.7362, "urban", True, 1500, 0.50, False, 4.0, 80),
        LocationConfig("Coimbatore",     11.0168, 76.9558, "urban", False),
        LocationConfig("Vadodara",       22.3072, 73.1812, "urban", False),
        LocationConfig("Patna",          25.6093, 85.1376, "urban", True, 1500, 0.40, False, 4.0, 80),
        LocationConfig("Bhubaneswar",    20.2961, 85.8245, "urban", True, 2000, 0.30, False, 5.0, 100),
        LocationConfig("Thiruvananthapuram", 8.5241, 76.9366, "coastal", False),
        LocationConfig("Surat",          21.1702, 72.8311, "urban", True, 2500, 0.35, False, 5.5, 130),
        LocationConfig("Kanpur",         26.4499, 80.3319, "urban", False),
        LocationConfig("Raipur",         21.2514, 81.6296, "urban", True, 1500, 0.30, False, 4.5, 90),
    ]

    # Build a lookup dict for haversine
    loc_dict = {loc.name: loc for loc in locations}

    # ── Routes ──────────────────────────────────────────────────────────
    route_defs = [
        # (src, tgt, terrain, road_grading, toll)
        # Delhi cluster
        ("Delhi", "Chandigarh",  "flat", 0.90, 250),
        ("Delhi", "Jaipur",      "flat", 0.85, 300),
        ("Delhi", "Lucknow",     "flat", 0.80, 350),
        ("Delhi", "Kanpur",      "flat", 0.80, 300),

        # Mumbai cluster
        ("Mumbai", "Pune",       "hilly", 0.90, 200),
        ("Mumbai", "Ahmedabad",  "flat", 0.85, 400),
        ("Mumbai", "Surat",      "coastal", 0.85, 250),
        ("Mumbai", "Nagpur",     "flat", 0.75, 500),

        # South India
        ("Chennai", "Bangalore", "flat", 0.85, 300),
        ("Chennai", "Hyderabad", "flat", 0.80, 400),
        ("Chennai", "Coimbatore", "hilly", 0.75, 250),
        ("Chennai", "Visakhapatnam", "coastal", 0.75, 350),

        # East India
        ("Kolkata", "Patna",       "flat", 0.75, 300),
        ("Kolkata", "Bhubaneswar", "flat", 0.80, 250),
        ("Kolkata", "Guwahati",    "hilly", 0.65, 400),

        # Bangalore cluster
        ("Bangalore", "Hyderabad", "flat", 0.85, 350),
        ("Bangalore", "Pune",      "hilly", 0.75, 500),
        ("Bangalore", "Coimbatore", "hilly", 0.80, 200),
        ("Bangalore", "Kochi",     "mountainous", 0.70, 300),

        # Central corridor
        ("Hyderabad", "Nagpur",       "flat", 0.80, 400),
        ("Hyderabad", "Visakhapatnam","coastal", 0.75, 300),

        # Central India
        ("Nagpur", "Bhopal",  "flat", 0.80, 300),
        ("Nagpur", "Raipur",  "flat", 0.80, 200),
        ("Bhopal", "Indore",  "flat", 0.85, 150),
        ("Bhopal", "Delhi",   "flat", 0.75, 500),
        ("Indore", "Ahmedabad", "flat", 0.80, 350),
        ("Indore", "Jaipur",    "hilly", 0.70, 400),

        # Gujarat corridor
        ("Ahmedabad", "Vadodara", "flat", 0.90, 100),
        ("Ahmedabad", "Jaipur",   "flat", 0.80, 400),
        ("Surat", "Vadodara",     "flat", 0.85, 100),

        # UP corridor
        ("Lucknow", "Kanpur", "flat", 0.85, 100),
        ("Lucknow", "Patna",  "flat", 0.70, 350),

        # East coast
        ("Visakhapatnam", "Bhubaneswar", "coastal", 0.75, 250),
        ("Bhubaneswar", "Raipur",        "hilly", 0.70, 300),

        # Kerala
        ("Kochi", "Coimbatore",          "mountainous", 0.70, 200),
        ("Kochi", "Thiruvananthapuram",  "coastal", 0.80, 150),
    ]

    routes: List[RouteConfig] = []
    for src, tgt, terrain, grading, toll in route_defs:
        routes.append(_make_route(src, tgt, loc_dict, terrain, grading, toll))

    # ── Vehicle fleet ───────────────────────────────────────────────────
    vehicles: List[VehicleConfig] = []
    truck_hubs = [
        "Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad",
        "Ahmedabad", "Pune", "Nagpur", "Lucknow",
    ]
    for i, hub in enumerate(truck_hubs):
        # Two trucks per major hub
        vehicles.append(VehicleConfig(
            f"truck_{i*2}", "truck", 10000, 5.0 + (i % 3), 2 + i % 5,
            2.0 + 0.5 * (i % 3), 55 + (i % 4) * 5, hub,
        ))
        vehicles.append(VehicleConfig(
            f"truck_{i*2+1}", "truck", 8000, 5.5 + (i % 2), 3 + i % 4,
            2.5, 60, hub,
        ))

    # A few rail vehicles on key corridors
    rail_hubs = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Nagpur"]
    for i, hub in enumerate(rail_hubs):
        vehicles.append(VehicleConfig(
            f"rail_{i}", "rail", 50000, 0.8, 5 + i, 1.0, 40, hub,
        ))

    # ── Shipment templates ─────────────────────────────────────────────
    templates = [
        ShipmentTemplate("electronics",   0.8, 720,  0.3, 400,  1.5, "high",     80000),
        ShipmentTemplate("pharmaceuticals",0.9, 240,  0.9, 200,  0.5, "critical", 200000),
        ShipmentTemplate("textiles",      0.2, 2160, 0.1, 1000, 5.0, "low",      20000),
        ShipmentTemplate("perishables",   0.6, 120,   1.0, 800,  3.0, "high",     40000),
        ShipmentTemplate("machinery",     0.4, 4320, 0.1, 5000, 10.0,"medium",   150000),
        ShipmentTemplate("fmcg",          0.3, 360,  0.2, 600,  2.5, "medium",   15000),
    ]

    cfg = ScenarioConfig(
        name="india_medium",
        locations=locations,
        routes=routes,
        vehicles=vehicles,
        shipment_templates=templates,
        max_steps=40,
    )
    cfg.auto_compute_times()
    return cfg
