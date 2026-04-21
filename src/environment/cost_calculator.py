"""
Cost calculator — computes all monetary costs for a shipment leg.

Covers:
- Fuel cost (distance, vehicle efficiency, fuel price)
- Toll cost (per-route)
- Maintenance cost (distance × rate)
- Mode-switch cost (when changing vehicle type)
- Insurance cost (cargo value × route risk)
- Mileage cost (per-km route cost)

All costs are in ₹ (Indian Rupees) for the demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config.default_config import (
    RouteConfig,
    VehicleConfig,
    ShipmentTemplate,
    TERRAIN_RISK_FACTOR,
    MODE_SWITCH_COST,
    BASE_FUEL_PRICE,
)


@dataclass
class LegCostBreakdown:
    """Itemized cost for one leg of the journey."""
    fuel_cost: float = 0.0
    toll_cost: float = 0.0
    maintenance_cost: float = 0.0
    mileage_cost: float = 0.0
    mode_switch_cost: float = 0.0
    insurance_cost: float = 0.0
    total: float = 0.0

    def compute_total(self):
        self.total = (
            self.fuel_cost
            + self.toll_cost
            + self.maintenance_cost
            + self.mileage_cost
            + self.mode_switch_cost
            + self.insurance_cost
        )
        return self.total


class CostCalculator:
    """Stateless cost computation for shipment legs."""

    def __init__(self, base_fuel_price: float = BASE_FUEL_PRICE):
        self.base_fuel_price = base_fuel_price

    def compute_leg_cost(
        self,
        route: RouteConfig,
        vehicle: VehicleConfig,
        shipment: ShipmentTemplate,
        anomaly_cost_factor: float = 1.0,
        fuel_price_factor: float = 1.0,
        previous_vehicle_type: Optional[str] = None,
        route_risk_score: float = 0.0,
    ) -> LegCostBreakdown:
        """
        Compute the full cost breakdown for traveling one route segment.

        Parameters
        ----------
        route : RouteConfig
            The route being traveled.
        vehicle : VehicleConfig
            The vehicle being used.
        shipment : ShipmentTemplate
            The cargo being transported.
        anomaly_cost_factor : float
            Multiplier from active anomalies (>= 1.0).
        fuel_price_factor : float
            Seasonal / temporal fuel price modulation.
        previous_vehicle_type : str or None
            If switching from a different vehicle type, mode-switch cost applies.
        route_risk_score : float
            Combined risk score for the route (0-1) from anomalies + terrain.
        """
        cost = LegCostBreakdown()
        distance = route.distance_km

        # ── Fuel cost ──────────────────────────────────────────────────
        fuel_price = self.base_fuel_price * fuel_price_factor
        if vehicle.fuel_efficiency_km_per_l > 0:
            litres = distance / vehicle.fuel_efficiency_km_per_l
            # Heavier loads consume more fuel (simplified linear model)
            load_factor = 1.0 + 0.3 * (shipment.weight_kg / vehicle.max_payload_kg)
            cost.fuel_cost = litres * fuel_price * load_factor
        else:
            # Electric / rail — minimal fuel cost
            cost.fuel_cost = distance * 0.5

        # ── Toll cost ──────────────────────────────────────────────────
        cost.toll_cost = route.toll_cost

        # ── Maintenance cost ───────────────────────────────────────────
        # Older vehicles cost more to maintain
        age_factor = 1.0 + 0.05 * vehicle.vehicle_age_years
        cost.maintenance_cost = distance * vehicle.maintenance_cost_per_km * age_factor

        # ── Mileage cost ───────────────────────────────────────────────
        cost.mileage_cost = distance * route.mileage_cost_per_km

        # ── Mode-switch cost ───────────────────────────────────────────
        if previous_vehicle_type and previous_vehicle_type != vehicle.vehicle_type:
            key = (previous_vehicle_type, vehicle.vehicle_type)
            cost.mode_switch_cost = MODE_SWITCH_COST.get(key, 5000.0)

        # ── Insurance cost ─────────────────────────────────────────────
        terrain_risk = TERRAIN_RISK_FACTOR.get(route.terrain, 0.1)
        combined_risk = min(terrain_risk + route_risk_score, 1.0)
        # Insurance is a fraction of cargo value scaled by risk
        cost.insurance_cost = shipment.insurance_value * combined_risk * 0.01

        # ── Apply anomaly multiplier to variable costs ─────────────────
        cost.fuel_cost *= anomaly_cost_factor
        cost.mileage_cost *= anomaly_cost_factor

        cost.compute_total()
        return cost
