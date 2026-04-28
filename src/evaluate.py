import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import torch
from tqdm import tqdm

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config.scenarios import india_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.ppo_agent import ActorCritic


def load_agent(env, device):
    """Load the best trained model checkpoint."""
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    model_path = os.path.join(ckpt_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        # Fallback to latest
        model_path = os.path.join(ckpt_dir, "latest_model.pt")
        if not os.path.exists(model_path):
            print("⚠ No trained model found. Trained agent results will be random.")
            return None

    metadata = (
        ['location', 'vehicle', 'shipment'],
        [('location', 'route', 'location'),
         ('vehicle', 'vehicle_at', 'location'),
         ('shipment', 'shipment_at', 'location'),
         ('shipment', 'shipment_dest', 'location'),
         ('location', 'rev_vehicle_at', 'vehicle'),
         ('location', 'rev_shipment_at', 'shipment'),
         ('location', 'rev_shipment_dest', 'shipment')]
    )

    agent = ActorCritic(
        metadata=metadata,
        hidden_channels=64,
        out_channels=64,
        max_neighbors=env.max_neighbors,
        max_vehicles=env.max_vehicles,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    print(f"✅ Loaded model from {os.path.basename(model_path)} (ep {checkpoint.get('episode', '?')})")
    return agent


def greedy_nominal_action(env):
    """Baseline: Pre-computed shortest path based on base time (ignores anomalies)."""
    try:
        path = nx.shortest_path(env.graph, env.current_node, env.destination, weight="base_time_hours")
        if len(path) > 1:
            next_node = path[1]
        else:
            next_node = env.current_node
    except nx.NetworkXNoPath:
        next_node = env._current_neighbors[0]

    if next_node in env._current_neighbors:
        next_hop_idx = env._current_neighbors.index(next_node)
    else:
        next_hop_idx = 0

    # Pick the fastest vehicle (assume index 0 or similar, or air/ship)
    # A simple heuristic: pick the first vehicle 
    vehicle_idx = 0 
    return np.array([next_hop_idx, vehicle_idx])


def oracle_dijkstra_action(env):
    """Oracle Baseline: Recomputes shortest path considering current active anomalies."""
    G = env.graph
    temp_G = nx.DiGraph()
    
    for u, v, data in G.edges(data=True):
        # Get base travel time from edge (already includes distance & terrain)
        base_time = data.get("base_time_hours", 1.0)
        # Apply current anomaly and traffic factors
        anomaly_factor = env.anomaly_engine.edge_time_factor(u, v)
        traffic_factor = env.time_engine.traffic_factor()
        node_factor = env.anomaly_engine.node_time_factor(v)
        weight = base_time * anomaly_factor * traffic_factor * node_factor
        temp_G.add_edge(u, v, weight=weight)
    
    try:
        path = nx.shortest_path(temp_G, env.current_node, env.destination, weight="weight")
        if len(path) > 1:
            next_node = path[1]
        else:
            next_node = env.current_node
    except nx.NetworkXNoPath:
        # Fallback to first available neighbor
        next_node = env._current_neighbors[0] if env._current_neighbors else env.current_node
    
    # Find action index that leads to next_node
    for i, (neighbor, vehicle_id) in enumerate(env.available_actions):
        if neighbor == next_node:
            return np.array([i, vehicle_id])
    # Ultimate fallback
    return np.array([0, 0])


def evaluate_policy(env, policy_name, agent, feature_engine, device, num_episodes):
    results = []
    
    # Run loop
    for ep in tqdm(range(num_episodes), desc=f"Evaluating {policy_name}"):
        obs, _ = env.reset(seed=1000 + ep) # Fixed seeds for fair comparison
        
        # Apply full difficulty curriculum for evaluation
        env.set_curriculum_phase(3, max_hops=50, allowed_vehicle_types=None)
        env.anomaly_engine.set_phase(3)
        
        done = False
        truncated = False
        ep_reward = 0.0
        
        while not (done or truncated):
            if policy_name == "Trained Agent":
                if agent is not None:
                    state_dict = env.get_graph_state()
                    state = feature_engine.build(state_dict).to(device)
                    with torch.no_grad():
                        # Deterministic action (mode)
                        action, _, _, _ = agent(state) 
                    a = action.cpu().numpy()
                else:
                    a = env.action_space.sample()
            elif policy_name == "Random":
                a = env.action_space.sample()
            elif policy_name == "Greedy Nominal":
                a = greedy_nominal_action(env)
            elif policy_name == "Oracle Dijkstra":
                a = oracle_dijkstra_action(env)
            
            obs, reward, done, truncated, _ = env.step(a)
            ep_reward += reward
            
        delivered = (env.current_node == env.destination)
        shelf_life_ratio = env.total_time_hours / max(1.0, env.shipment.shelf_life_hours)
        success = delivered and (shelf_life_ratio <= 1.0)
        
        anomalies_encountered = sum(len(leg.get("anomalies", [])) for leg in env.leg_details)
        
        results.append({
            "Policy": policy_name,
            "Episode": ep,
            "Reward": ep_reward,
            "Delivered": delivered,
            "Success": success,
            "Steps": env.step_count,
            "Total_Time_hrs": env.total_time_hours,
            "Total_Cost": env.total_cost,
            "Total_Risk": env.total_risk,
            "Anomalies_Hit": anomalies_encountered
        })
        
    return results


def main():
    print("🚀 Starting Evaluation Suite...")
    num_episodes = 100
    
    config = india_scenario()
    env = SupplyChainEnv(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_engine = FeatureEngine()
    agent = load_agent(env, device)
    
    all_results = []
    
    policies = ["Trained Agent", "Greedy Nominal", "Oracle Dijkstra", "Random"]
    
    for policy in policies:
        res = evaluate_policy(env, policy, agent, feature_engine, device, num_episodes)
        all_results.extend(res)
        
    df = pd.DataFrame(all_results)
    
    # Save raw results
    out_file = "evaluation_results.csv"
    df.to_csv(out_file, index=False)
    print(f"\n✅ Evaluation complete. Saved results to {out_file}\n")
    
    # Print summary
    summary = df.groupby("Policy").agg({
        "Success": "mean",
        "Reward": ["mean", "std"],
        "Total_Time_hrs": "mean",
        "Total_Cost": "mean",
        "Steps": "mean",
        "Anomalies_Hit": "mean"
    }).round(2)
    
    # Rename columns for cleaner print
    summary.columns = ["Success Rate", "Reward (Mean)", "Reward (Std)", "Time (hrs)", "Cost (₹)", "Steps", "Anomalies Hit"]
    summary["Success Rate"] = (summary["Success Rate"] * 100).astype(str) + "%"
    
    print("📊 PERFORMANCE SUMMARY (100 Episodes):")
    print("-" * 100)
    print(summary.to_string())
    print("-" * 100)


if __name__ == "__main__":
    main()
