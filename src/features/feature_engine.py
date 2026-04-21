"""
Feature engine — converts environment state into PyG HeteroData.

This is the bridge between the Gymnasium environment (flat observations)
and the GNN encoder (heterogeneous graph with typed nodes/edges).

Node types: location, vehicle, shipment
Edge types: route (location→location), vehicle_at (vehicle→location),
            shipment_at (shipment→location), shipment_dest (shipment→location)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from src.config.default_config import (
    ScenarioConfig,
    ShipmentTemplate,
    VehicleConfig,
    TERRAIN_SPEED_FACTOR,
    TERRAIN_RISK_FACTOR,
    VEHICLE_TYPE_INDEX,
)
from src.environment.anomaly_engine import AnomalyEngine
from src.environment.time_engine import TimeEngine


class FeatureEngine:
    """
    Builds a torch_geometric HeteroData object from environment state.

    Call build() with the dict returned by env.get_graph_state().
    """

    def __init__(self):
        # Encoding lookups
        self._terrain_map = {"flat": 0.2, "hilly": 0.5, "mountainous": 0.8, "coastal": 0.4}
        self._region_map = {
            "metro": 1.0, "urban": 0.7, "hub": 0.8,
            "port": 0.6, "rural": 0.3, "coastal": 0.5,
        }
        self._priority_map = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
        self._vtype_map = {"truck": 0.25, "rail": 0.5, "air": 0.75, "ship": 1.0}

    def build(self, state: dict) -> HeteroData:
        """
        Build HeteroData from the environment's graph state dict.

        Parameters
        ----------
        state : dict
            The dict returned by env.get_graph_state(), containing:
            config, graph, current_node, destination, shipment,
            vehicles, vehicle_positions, anomaly_engine, time_engine,
            step_count, total_time_hours.
        """
        config: ScenarioConfig = state["config"]
        graph = state["graph"]
        current_node: str = state["current_node"]
        destination: str = state["destination"]
        shipment: ShipmentTemplate = state["shipment"]
        vehicles: List[VehicleConfig] = state["vehicles"]
        vehicle_positions: Dict[str, str] = state["vehicle_positions"]
        anomaly_engine: AnomalyEngine = state["anomaly_engine"]
        time_engine: TimeEngine = state["time_engine"]
        step_count: int = state["step_count"]
        total_time_hours: float = state["total_time_hours"]

        data = HeteroData()
        loc_names = config.location_names()
        loc_idx = {name: i for i, name in enumerate(loc_names)}

        # ══════════════════════════════════════════════════════════════
        # Location node features
        # ══════════════════════════════════════════════════════════════
        loc_features = []
        for loc in config.locations:
            risk = anomaly_engine.node_risk_score(loc.name)
            loc_features.append([
                loc.lat / 35.0,
                loc.lng / 100.0,
                risk,
                self._region_map.get(loc.region_type, 0.5),
                float(loc.has_warehouse),
                loc.warehouse_fill_ratio if loc.has_warehouse else 0.0,
                float(loc.name == current_node),
                float(loc.name == destination),
            ])
        data["location"].x = torch.tensor(loc_features, dtype=torch.float)

        # ══════════════════════════════════════════════════════════════
        # Route edge features + edge_index
        # ══════════════════════════════════════════════════════════════
        edge_src, edge_dst, edge_features = [], [], []
        for u, v, edata in graph.edges(data=True):
            if u not in loc_idx or v not in loc_idx:
                continue
            edge_src.append(loc_idx[u])
            edge_dst.append(loc_idx[v])

            anomaly_time = anomaly_engine.edge_time_factor(u, v)
            anomaly_cost = anomaly_engine.edge_cost_factor(u, v)

            edge_features.append([
                edata["distance_km"] / 1500.0,
                self._terrain_map.get(edata["terrain"], 0.5),
                edata["road_grading"],
                edata["toll_cost"] / 1000.0,
                edata["mileage_cost_per_km"] / 10.0,
                edata["base_time_hours"] / 20.0,
                anomaly_time / 3.0,
                anomaly_cost / 3.0,
                TERRAIN_RISK_FACTOR.get(edata["terrain"], 0.1),
                float(v == destination),
            ])

        if edge_src:
            data["location", "route", "location"].edge_index = torch.tensor(
                [edge_src, edge_dst], dtype=torch.long
            )
            data["location", "route", "location"].edge_attr = torch.tensor(
                edge_features, dtype=torch.float
            )

        # ══════════════════════════════════════════════════════════════
        # Vehicle node features
        # ══════════════════════════════════════════════════════════════
        vehicle_features = []
        veh_at_src, veh_at_dst = [], []

        for i, v in enumerate(vehicles):
            vehicle_features.append([
                self._vtype_map.get(v.vehicle_type, 0.25),
                v.max_payload_kg / 50000.0,
                v.fuel_efficiency_km_per_l / 10.0,
                v.vehicle_age_years / 15.0,
                v.maintenance_cost_per_km / 5.0,
                v.speed_kmph / 100.0,
                float(shipment.weight_kg <= v.max_payload_kg),
            ])

            # vehicle_at edge: vehicle → its current location
            pos = vehicle_positions.get(v.vehicle_id, v.home_location)
            if pos in loc_idx:
                veh_at_src.append(i)
                veh_at_dst.append(loc_idx[pos])

        data["vehicle"].x = torch.tensor(vehicle_features, dtype=torch.float)

        if veh_at_src:
            data["vehicle", "vehicle_at", "location"].edge_index = torch.tensor(
                [veh_at_src, veh_at_dst], dtype=torch.long
            )

        # ══════════════════════════════════════════════════════════════
        # Shipment node features (single node)
        # ══════════════════════════════════════════════════════════════
        remaining_shelf = max(
            0, 1.0 - total_time_hours / shipment.shelf_life_hours
        )
        density = shipment.weight_kg / max(shipment.volume_m3, 0.01)

        shipment_features = [[
            hash(shipment.product_type) % 100 / 100.0,
            shipment.fragility,
            min(shipment.shelf_life_hours / 1000.0, 5.0),
            shipment.temperature_sensitivity,
            shipment.weight_kg / 10000.0,
            shipment.volume_m3 / 20.0,
            density / 1000.0,
            shipment.insurance_value / 200000.0,
            self._priority_map.get(shipment.priority, 0.5),
            remaining_shelf,
        ]]
        data["shipment"].x = torch.tensor(shipment_features, dtype=torch.float)

        # shipment_at edge: shipment → current location
        if current_node in loc_idx:
            data["shipment", "shipment_at", "location"].edge_index = torch.tensor(
                [[0], [loc_idx[current_node]]], dtype=torch.long
            )

        # shipment_dest edge: shipment → destination
        if destination in loc_idx:
            data["shipment", "shipment_dest", "location"].edge_index = torch.tensor(
                [[0], [loc_idx[destination]]], dtype=torch.long
            )

        # ══════════════════════════════════════════════════════════════
        # Global context (stored as graph-level attributes)
        # ══════════════════════════════════════════════════════════════
        ctx = time_engine.get_context_vector()
        data.global_context = torch.tensor([ctx], dtype=torch.float)
        data.step_progress = torch.tensor(
            [[step_count / 40.0, total_time_hours / max(shipment.shelf_life_hours, 1)]],
            dtype=torch.float,
        )

        # Attach indices for action selection in the Actor network
        if current_node in loc_idx:
            data.current_node_idx = torch.tensor([loc_idx[current_node]], dtype=torch.long)
        else:
            data.current_node_idx = torch.tensor([0], dtype=torch.long)
            
        # The environment uses env._current_neighbors for the action space index
        # We need to map those neighbors to their location node indices
        # state["neighbors"] is not passed to get_graph_state yet. 
        # Wait, get_graph_state() doesn't have neighbors. I'll get neighbors from graph directly.
        from src.utils.graph_utils import get_neighbors
        neighbors = get_neighbors(graph, current_node)
        neighbor_indices = [loc_idx[n] for n in neighbors if n in loc_idx]
        data.neighbor_indices = torch.tensor(neighbor_indices, dtype=torch.long)

        # Make the graph undirected so vehicles and shipments receive messages from locations
        # (adds rev_route, rev_vehicle_at, etc.)
        data = T.ToUndirected()(data)

        return data
