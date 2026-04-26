"""
Anomaly engine — stochastic disruption simulation.

Simulates four disruption types (weather, traffic, sentiment, geopolitical)
on edges and/or nodes. Each anomaly appears/disappears probabilistically
and carries a severity multiplier that affects travel time and cost.

Designed with a clean interface so real data sources (weather APIs, news
sentiment models) can replace the stochastic generators later.

Supports dynamic spawn probability scaling for curriculum learning:
  engine.set_phase(1)          # Phase 1 — 30% of base spawn rates
  engine.set_spawn_scale(0.6)  # Or set any scale directly
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from src.config.default_config import AnomalyConfig, AnomalyTypeConfig


@dataclass
class ActiveAnomaly:
    """A currently active disruption."""
    anomaly_type: str           # weather | traffic | sentiment | geopolitical
    severity: float             # multiplier on travel time (>= 1.0)
    cost_multiplier: float      # multiplier on cost
    ticks_active: int = 0       # how many steps it has been active


class AnomalyEngine:
    """
    Manages disruption state across the entire graph.

    State is tracked per-edge and per-node, with each key potentially
    having multiple concurrent anomalies of different types.
    """

    # Phase-to-scale mapping for curriculum learning
    _PHASE_SCALE = {1: 0.3, 2: 0.6, 3: 1.0}

    def __init__(self, config: AnomalyConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.rng = rng or random.Random()

        # Active anomalies keyed by location name or (src, tgt) edge tuple
        self.edge_anomalies: Dict[Tuple[str, str], List[ActiveAnomaly]] = {}
        self.node_anomalies: Dict[str, List[ActiveAnomaly]] = {}

        # ── Curriculum scaling ─────────────────────────────────────────
        # Scale factor applied to all prob_appear values (0.0–1.0).
        # Allows the training loop to reduce anomaly frequency for
        # early curriculum phases, then ramp up to full difficulty.
        self._spawn_scale: float = 1.0

    def set_spawn_scale(self, scale: float) -> None:
        """
        Set the spawn-probability scale factor (curriculum learning).

        Parameters
        ----------
        scale : float
            Multiplier in [0.0, 1.0] applied to every anomaly type's
            ``prob_appear_per_step``.  A value of 0.3 means anomalies
            spawn at 30% of their configured base rate.
        """
        self._spawn_scale = max(0.0, min(float(scale), 1.0))

    def get_spawn_scale(self) -> float:
        """Return the current spawn-probability scale factor."""
        return self._spawn_scale

    def set_phase(self, phase: int) -> None:
        """
        Convenience wrapper: set anomaly difficulty by curriculum phase.

        Phase 1 → 30% of base spawn rates  (easy)
        Phase 2 → 60% of base spawn rates  (medium)
        Phase 3 → 100% of base spawn rates (full difficulty)
        """
        scale = self._PHASE_SCALE.get(phase, 1.0)
        self.set_spawn_scale(scale)

    def _anomaly_types(self) -> Dict[str, AnomalyTypeConfig]:
        """Return all anomaly type configs as a dict."""
        return {
            "weather": self.config.weather,
            "traffic": self.config.traffic,
            "sentiment": self.config.sentiment,
            "geopolitical": self.config.geopolitical,
        }

    def initialize(
        self,
        edge_keys: List[Tuple[str, str]],
        node_keys: List[str],
        warmup_steps: int = 5,
    ):
        """
        Initialize anomaly state for all edges and nodes.
        Run a few warmup steps so the initial state isn't all-clean.
        """
        self.edge_anomalies = {k: [] for k in edge_keys}
        self.node_anomalies = {k: [] for k in node_keys}

        for _ in range(warmup_steps):
            self.step()

    def step(self):
        """Advance anomaly state by one tick: spawn new, expire old."""
        type_configs = self._anomaly_types()

        for atype, cfg in type_configs.items():
            if cfg.affects in ("edges", "both"):
                self._update_dict(self.edge_anomalies, atype, cfg)
            if cfg.affects in ("nodes", "both"):
                self._update_dict(self.node_anomalies, atype, cfg)

    def _update_dict(
        self,
        anomaly_dict: Dict,
        atype: str,
        cfg: AnomalyTypeConfig,
    ):
        """Spawn / expire anomalies for one dict (edges or nodes)."""
        # Apply curriculum scale to the spawn probability only —
        # disappear probability stays unchanged so existing anomalies
        # clear at the normal rate.
        scaled_appear = cfg.prob_appear_per_step * self._spawn_scale

        for key, active_list in anomaly_dict.items():
            # --- Spawn new ---
            already_has = any(a.anomaly_type == atype for a in active_list)
            if not already_has and self.rng.random() < scaled_appear:
                severity = self.rng.uniform(cfg.severity_min, cfg.severity_max)
                active_list.append(ActiveAnomaly(
                    anomaly_type=atype,
                    severity=severity,
                    cost_multiplier=cfg.cost_multiplier,
                    ticks_active=0,
                ))

            # --- Expire existing ---
            to_remove = []
            for i, anom in enumerate(active_list):
                if anom.anomaly_type == atype:
                    anom.ticks_active += 1
                    if self.rng.random() < cfg.prob_disappear_per_step:
                        to_remove.append(i)
            for i in reversed(to_remove):
                active_list.pop(i)

    # ── Query methods ──────────────────────────────────────────────────

    def edge_time_factor(self, src: str, tgt: str) -> float:
        """Combined time multiplier for an edge (product of all severities)."""
        anomalies = self.edge_anomalies.get((src, tgt), [])
        factor = 1.0
        for a in anomalies:
            factor *= a.severity
        return factor

    def edge_cost_factor(self, src: str, tgt: str) -> float:
        """Combined cost multiplier for an edge."""
        anomalies = self.edge_anomalies.get((src, tgt), [])
        factor = 1.0
        for a in anomalies:
            factor *= a.cost_multiplier
        return factor

    def node_risk_score(self, node: str) -> float:
        """Risk score for a node based on active anomalies (0-1 scale)."""
        anomalies = self.node_anomalies.get(node, [])
        if not anomalies:
            return 0.0
        # Aggregate: more anomalies + higher severity = higher risk
        total = sum(a.severity - 1.0 for a in anomalies)
        return min(total / 3.0, 1.0)  # clamp to [0, 1]

    def node_time_factor(self, node: str) -> float:
        """Time multiplier from node-level anomalies (e.g. port delays)."""
        anomalies = self.node_anomalies.get(node, [])
        factor = 1.0
        for a in anomalies:
            factor *= a.severity
        return factor

    def get_edge_anomaly_summary(self, src: str, tgt: str) -> List[dict]:
        """Human-readable summary of active anomalies on an edge."""
        return [
            {"type": a.anomaly_type, "severity": a.severity, "ticks": a.ticks_active}
            for a in self.edge_anomalies.get((src, tgt), [])
        ]

    def get_node_anomaly_summary(self, node: str) -> List[dict]:
        """Human-readable summary of active anomalies on a node."""
        return [
            {"type": a.anomaly_type, "severity": a.severity, "ticks": a.ticks_active}
            for a in self.node_anomalies.get(node, [])
        ]

    def get_all_active_count(self) -> int:
        """Total number of active anomalies across the graph."""
        total = sum(len(v) for v in self.edge_anomalies.values())
        total += sum(len(v) for v in self.node_anomalies.values())
        return total
