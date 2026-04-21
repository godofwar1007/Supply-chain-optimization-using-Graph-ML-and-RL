"""
Graph utilities — build and query the supply chain network.

Uses NetworkX as the canonical graph representation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import networkx as nx

from src.config.default_config import ScenarioConfig, RouteConfig


def compute_haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_networkx_graph(config: ScenarioConfig) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a ScenarioConfig.

    Nodes carry location attributes.
    Edges carry route attributes.
    Bidirectional routes create two directed edges.
    """
    G = nx.DiGraph()

    # Add location nodes
    for loc in config.locations:
        G.add_node(loc.name, **{
            "lat": loc.lat,
            "lng": loc.lng,
            "region_type": loc.region_type,
            "has_warehouse": loc.has_warehouse,
            "warehouse_capacity": loc.warehouse_capacity,
            "warehouse_fill_ratio": loc.warehouse_fill_ratio,
            "cold_storage": loc.cold_storage,
            "handling_cost": loc.handling_cost,
            "throughput_rate": loc.throughput_rate,
        })

    # Add route edges
    for route in config.routes:
        edge_attrs = {
            "distance_km": route.distance_km,
            "terrain": route.terrain,
            "road_grading": route.road_grading,
            "base_time_hours": route.base_time_hours,
            "toll_cost": route.toll_cost,
            "mileage_cost_per_km": route.mileage_cost_per_km,
        }
        G.add_edge(route.source, route.target, **edge_attrs)
        if route.bidirectional:
            G.add_edge(route.target, route.source, **edge_attrs)

    return G


def get_neighbors(G: nx.DiGraph, node: str) -> List[str]:
    """Return list of successor node names."""
    return list(G.successors(node))


def get_route_config(
    config: ScenarioConfig, src: str, tgt: str
) -> RouteConfig:
    """Find the RouteConfig for a src→tgt edge."""
    for route in config.routes:
        if route.source == src and route.target == tgt:
            return route
        if route.bidirectional and route.source == tgt and route.target == src:
            return route
    raise ValueError(f"No route found for {src} → {tgt}")


def shortest_path_distance(G: nx.DiGraph, src: str, tgt: str) -> float:
    """Shortest path distance in km (Dijkstra)."""
    try:
        path = nx.shortest_path(G, src, tgt, weight="distance_km")
        dist = 0.0
        for i in range(len(path) - 1):
            dist += G[path[i]][path[i + 1]]["distance_km"]
        return dist
    except nx.NetworkXNoPath:
        return float("inf")


def get_all_edge_keys(config: ScenarioConfig) -> List[Tuple[str, str]]:
    """Return all directed edge keys, expanding bidirectional routes."""
    keys = []
    for route in config.routes:
        keys.append((route.source, route.target))
        if route.bidirectional:
            keys.append((route.target, route.source))
    return keys
