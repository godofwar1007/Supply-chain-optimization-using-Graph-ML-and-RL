import gymnasium as gym 
from gymnasium import spaces
import numpy as np 
import random
from typing import Dict,List

class supply_chain_env_fixed_route(gym.Env):

    def __init__(self, config=None):
        super().__init__()

        self.config = config or self.default_config()
        self.graph_builder()

        self.max_parallel = max(len(edges) for edges in self.segment_edges.values())
        self.action_space = spaces.Discrete(self.max_parallel)

        # observation space: normalized step + anomaly factor for each possible edge
        obs_dim = 1 + self.max_parallel
        self.observation_space = spaces.Box(low=0, high=5.0, shape=(obs_dim,), dtype=np.float32)

    def default_config(self):
        return {
        "nodes": ["A", "B", "C", "D"],
        "route": ["A", "B", "C", "D"],
        "edges": [
            # A-B (2 edges)
            {"source": "A", "target": "B", "edge_id": 0, "distance_km": 50, "terrain": "flat", "base_time_min": 30},
            {"source": "A", "target": "B", "edge_id": 1, "distance_km": 70, "terrain": "hilly", "base_time_min": 45},
            # A-C
            {"source": "A", "target": "C", "edge_id": 0, "distance_km": 90, "terrain": "mountainous", "base_time_min": 80},
            {"source": "A", "target": "C", "edge_id": 1, "distance_km": 110, "terrain": "hilly", "base_time_min": 95},
            # A-D
            {"source": "A", "target": "D", "edge_id": 0, "distance_km": 140, "terrain": "flat", "base_time_min": 120},
            {"source": "A", "target": "D", "edge_id": 1, "distance_km": 160, "terrain": "mountainous", "base_time_min": 150},
            # B-C (2 edges)
            {"source": "B", "target": "C", "edge_id": 0, "distance_km": 40, "terrain": "flat", "base_time_min": 25},
            {"source": "B", "target": "C", "edge_id": 1, "distance_km": 60, "terrain": "mountainous", "base_time_min": 50},
            # B-D
            {"source": "B", "target": "D", "edge_id": 0, "distance_km": 100, "terrain": "hilly", "base_time_min": 85},
            {"source": "B", "target": "D", "edge_id": 1, "distance_km": 120, "terrain": "flat", "base_time_min": 105},
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

        self.segment_edges={}

        for edge in self.config["edges"]:

            key = (edge["source"],edge["target"])
            if key not in self.segment_edges:
                self.segment_edges[key]=[]
            self.segment_edges[key].append(edge)

        # just sorting by edge id (moved outside the loop)
        for key in self.segment_edges:
            self.segment_edges[key].sort(key=lambda e:e["edge_id"])

        # building the list of segments in order 
        self.segments = []
        route = self.config["route"]
        for i in range(len(route)-1):
            src,tgt = route[i],route[i+1]
            if (src,tgt) not in self.segment_edges:
                raise ValueError(f"Missing edges for segment {src} -> {tgt}")
            self.segments.append((src,tgt))
        self.num_segments=len(self.segments)
    
    def init_anomalies(self):

        self.anomaly_state = {}
        
        for (src,tgt), edges in self.segment_edges.items():  # fixed typo: segment_edges
            for edge in edges:
                key = (src,tgt,edge["edge_id"])
                self.anomaly_state[key]=[]

    def update_anomalies(self):

        anomaly_cfg = self.config["anomaly_config"]
        for key,active_list in self.anomaly_state.items():

            for anomaly_name,cfg in anomaly_cfg.items():

                if random.random() < cfg["prob_appear_per_step"]:

                    if not any( a['name'] == anomaly_name for a in active_list):
                        active_list.append({"name" : anomaly_name, "multiplier" : cfg["multiplier"]})

            to_remove = []
            for idx,anom in enumerate(active_list):
                cfg = anomaly_cfg[anom['name']]
                if random.random() < cfg["prob_disappear_per_step"]:
                    to_remove.append(idx)
            # moved outside the inner loop
            for idx in reversed(to_remove):
                active_list.pop(idx)

    def _get_current_segment_edges(self) -> List[Dict]:
        """Return list of edges for the current segment."""
        src, tgt = self.segments[self.step_idx]
        return self.segment_edges[(src, tgt)]                   
    
    def anomaly_factor(self, src: str, tgt: str, edge_id: int) -> float:

        # this is the function made to compute the anomaly factor 

        key = (src, tgt, edge_id)
        active = self.anomaly_state.get(key, [])
        if not active:
            return 1.0
        product = 1.0

        for anom in active:
            product *= anom["multiplier"]   # fixed typo: multipler -> multiplier

        return product

    def travel_time(self, edge: Dict, anomaly_factor: float) -> float:
           
        terrain_mult = self.config["terrain_multiplier"][edge["terrain"]]
        distance = edge["distance_km"]
        base_time = edge["base_time_min"]
        noise = np.random.lognormal(0, self.config["noise_std"])  # multiplicative noise

        travel_time = base_time * terrain_mult * anomaly_factor * noise
        return travel_time
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.total_time = 0.0
        self.init_anomalies()        # fixed: no underscore
        # Initial random anomalies
        for _ in range(5):  # adding some initial anomalies randomly
            self.update_anomalies()  # fixed: no underscore
        obs = self.get_obs()
        return obs, {}

    def get_obs(self) -> np.ndarray:

        norm_step = self.step_idx / self.num_segments
        obs = [norm_step]
        edges = self._get_current_segment_edges()

        edge_anomaly = {}
        for edge in edges:
            # fixed: call anomaly_factor correctly
            factor = self.anomaly_factor(edge["source"], edge["target"], edge["edge_id"])
            edge_anomaly[edge["edge_id"]] = factor 

        for edge_id in range(self.max_parallel):  # fill in for all possible edge indices 
            if edge_id in edge_anomaly:
                obs.append(edge_anomaly[edge_id])
            else:
                obs.append(1.0)
        return np.array(obs, dtype=np.float32)
                        
    def step(self, action: int):
        edges = self._get_current_segment_edges()
        chosen_edge = None

        for e in edges:
            if e["edge_id"] == action:
                chosen_edge = e
                break
        if chosen_edge is None:

            # this is an invalid action as the agent should not remain in the same step so large penalty
            # so we will give a huge time penalty and move to the next step 

            travel_time = 1e6
            reward = -travel_time
            self.total_time += travel_time   # added: update total time
            self.step_idx += 1
            done = self.step_idx >= self.num_segments   # fixed: >=
            self.update_anomalies()

            obs = np.zeros(self.observation_space.shape) if done else self.get_obs()
            return obs, reward, done, False, {"total_time": self.total_time, "invalid_action": True}   
        
        # valid action 
        anomaly_factor = self.anomaly_factor(chosen_edge["source"], chosen_edge["target"], action)
        travel_time = self.travel_time(chosen_edge, anomaly_factor)   # fixed: call travel_time
        self.total_time += travel_time
        reward = -travel_time
        
        self.step_idx += 1
        done = self.step_idx >= self.num_segments

        # update the anomalies for the next step 
        self.update_anomalies()

        if done:
           obs = np.zeros(self.observation_space.shape) if done else self.get_obs()
        else:
            obs = self.get_obs()

        info = {"total_time": self.total_time, "chosen_edge_id": action, "anomaly_factor": anomaly_factor}
        return obs, reward, done, False, info
    
    def render(self, mode='human'):
        if mode == 'human' and self.step_idx < self.num_segments:
            src, tgt = self.segments[self.step_idx]
            print(f"Step {self.step_idx}/{self.num_segments} | Segment {src}->{tgt} | Total time: {self.total_time:.2f} min")
