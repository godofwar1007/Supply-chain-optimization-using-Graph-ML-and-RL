import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import random
from typing import Dict, List

class supply_chain_env_fixed_route(gym.Env):

    def __init__(self, config=None):
        super().__init__()

        self.config = config or self.default_config()
        self.graph_builder()

        self.max_parallel = max(len(edges) for edges in self.segment_edges.values())
        self.action_space = spaces.Discrete(self.max_parallel)

        # observation space: norm_step + (base_time_norm, anomaly_factor) for each possible edge
        obs_dim = 1 + 2 * self.max_parallel
        self.observation_space = spaces.Box(low=0, high=10.0, shape=(obs_dim,), dtype=np.float32)

    def default_config(self):
        return {
            "nodes": ["A", "B", "C", "D"],
            "route": ["A", "B", "C", "D"],
            "edges": [
                # A-B (2 edges)
                {"source": "A", "target": "B", "edge_id": 0, "distance_km": 50, "terrain": "flat", "base_time_min": 30},
                {"source": "A", "target": "B", "edge_id": 1, "distance_km": 70, "terrain": "hilly", "base_time_min": 45},
                # B-C (2 edges)
                {"source": "B", "target": "C", "edge_id": 0, "distance_km": 40, "terrain": "flat", "base_time_min": 25},
                {"source": "B", "target": "C", "edge_id": 1, "distance_km": 60, "terrain": "mountainous", "base_time_min": 50},
                # C-D (2 edges)
                {"source": "C", "target": "D", "edge_id": 0, "distance_km": 30, "terrain": "flat", "base_time_min": 20},
                {"source": "C", "target": "D", "edge_id": 1, "distance_km": 55, "terrain": "hilly", "base_time_min": 38},
            ],
            "anomaly_config": {
                "weather": {"prob_appear_per_step": 0.1, "prob_disappear_per_step": 0.2, "multiplier": 1.5},
                "geopolitical": {"prob_appear_per_step": 0.05, "prob_disappear_per_step": 0.1, "multiplier": 2.0}
            },
            "terrain_multiplier": {"flat": 1.0, "hilly": 1.2, "mountainous": 1.5},
            "noise_std": 0.05
        }

    def graph_builder(self):
        self.segment_edges = {}
        for edge in self.config["edges"]:
            key = (edge["source"], edge["target"])
            if key not in self.segment_edges:
                self.segment_edges[key] = []
            self.segment_edges[key].append(edge)

        for key in self.segment_edges:
            self.segment_edges[key].sort(key=lambda e: e["edge_id"])

        self.segments = []
        route = self.config["route"]
        for i in range(len(route) - 1):
            src, tgt = route[i], route[i+1]
            if (src, tgt) not in self.segment_edges:
                raise ValueError(f"Missing edges for segment {src} -> {tgt}")
            self.segments.append((src, tgt))
        self.num_segments = len(self.segments)
    
    def init_anomalies(self):
        self.anomaly_state = {}
        for (src, tgt), edges in self.segment_edges.items():
            for edge in edges:
                key = (src, tgt, edge["edge_id"])
                self.anomaly_state[key] = []

    def update_anomalies(self):
        anomaly_cfg = self.config["anomaly_config"]
        for key, active_list in self.anomaly_state.items():
            for anomaly_name, cfg in anomaly_cfg.items():
                if random.random() < cfg["prob_appear_per_step"]:
                    if not any(a['name'] == anomaly_name for a in active_list):
                        active_list.append({"name": anomaly_name, "multiplier": cfg["multiplier"]})

            to_remove = []
            for idx, anom in enumerate(active_list):
                cfg = anomaly_cfg[anom['name']]
                if random.random() < cfg["prob_disappear_per_step"]:
                    to_remove.append(idx)
            for idx in reversed(to_remove):
                active_list.pop(idx)

    def _get_current_segment_edges(self) -> List[Dict]:
        """Return list of edges for the current segment."""
        src, tgt = self.segments[self.step_idx]
        return self.segment_edges[(src, tgt)]                   
    
    def anomaly_factor(self, src: str, tgt: str, edge_id: int) -> float:
        key = (src, tgt, edge_id)
        active = self.anomaly_state.get(key, [])
        if not active:
            return 1.0
        product = 1.0
        for anom in active:
            product *= anom["multiplier"]
        return product

    def travel_time(self, edge: Dict, anomaly_factor: float) -> float:
        terrain_mult = self.config["terrain_multiplier"][edge["terrain"]]
        base_time = edge["base_time_min"]
        noise = np.random.lognormal(0, self.config["noise_std"])
        travel_time = base_time * terrain_mult * anomaly_factor * noise
        return travel_time
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.total_time = 0.0
        self.init_anomalies()
        for _ in range(5):
            self.update_anomalies()
        obs = self.get_obs()
        return obs, {}

    def get_obs(self) -> np.ndarray:
        norm_step = self.step_idx / self.num_segments
        obs = [norm_step]
        edges = self._get_current_segment_edges()

        edge_info = {}
        for edge in edges:
            factor = self.anomaly_factor(edge["source"], edge["target"], edge["edge_id"])
            # normalized base time (including terrain mult)
            base_time_norm = (edge["base_time_min"] * self.config["terrain_multiplier"][edge["terrain"]]) / 100.0
            edge_info[edge["edge_id"]] = (base_time_norm, factor)

        for edge_id in range(self.max_parallel):
            if edge_id in edge_info:
                base, anomaly = edge_info[edge_id]
                obs.extend([base, anomaly])
            else:
                obs.extend([0.0, 1.0])
        return np.array(obs, dtype=np.float32)
                        
    def step(self, action: int):
        edges = self._get_current_segment_edges()
        chosen_edge = None

        for e in edges:
            if e["edge_id"] == action:
                chosen_edge = e
                break
        if chosen_edge is None:
            travel_time = 1e6
            reward = -travel_time
            self.total_time += travel_time
            self.step_idx += 1
            done = self.step_idx >= self.num_segments
            self.update_anomalies()
            obs = np.zeros(self.observation_space.shape) if done else self.get_obs()
            return obs, reward, done, False, {"total_time": self.total_time, "invalid_action": True}   
        
        anomaly_factor = self.anomaly_factor(chosen_edge["source"], chosen_edge["target"], action)
        travel_time = self.travel_time(chosen_edge, anomaly_factor)
        self.total_time += travel_time
        reward = -travel_time
        
        self.step_idx += 1
        done = self.step_idx >= self.num_segments
        self.update_anomalies()

        if done:
           obs = np.zeros(self.observation_space.shape)
        else:
            obs = self.get_obs()

        info = {"total_time": self.total_time, "chosen_edge_id": action, "anomaly_factor": anomaly_factor}
        return obs, reward, done, False, info
    
    def render(self, mode='human'):
        if mode == 'human':
            if self.step_idx >= self.num_segments:
                print(f"\n\033[92m[COMPLETED]\033[0m Final Time: {self.total_time:.2f} min")
                return

            src, tgt = self.segments[self.step_idx]
            edges = self._get_current_segment_edges()
            
            print(f"\n\033[1;34m--- Step {self.step_idx + 1}/{self.num_segments}: {src} \u2192 {tgt} ---\033[0m")
            print(f"{'Edge ID':<10} | {'Base Time':<10} | {'Terrain':<12} | {'Anomalies':<15}")
            print("-" * 55)
            
            for edge in edges:
                factor = self.anomaly_factor(edge["source"], edge["target"], edge["edge_id"])
                color = "\033[91m" if factor > 1.1 else "\033[92m"
                anom_str = f"{color}{factor:.2f}x\033[0m"
                
                base_time = edge["base_time_min"] * self.config["terrain_multiplier"][edge["terrain"]]
                print(f"{edge['edge_id']:<10} | {base_time:<10.1f} | {edge['terrain']:<12} | {anom_str}")
            print(f"\033[90mTotal Accumulated Time: {self.total_time:.2f} min\033[0m")
