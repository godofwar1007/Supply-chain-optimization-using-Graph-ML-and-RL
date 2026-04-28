import os
import sys
import networkx as nx
import torch

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config.scenarios import india_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.evaluate import load_agent
from src.environment.anomaly_engine import ActiveAnomaly

def main():
    print("🚚 Starting Reroute Test...")
    
    config = india_scenario()
    env = SupplyChainEnv(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    feature_engine = FeatureEngine()
    agent = load_agent(env, device)
    
    # Force a specific origin and destination
    origin = "Delhi"
    destination = "Chennai"
    
    # We must reset first to initialize env state, then override
    obs, _ = env.reset(seed=42)
    env.current_node = origin
    env.destination = destination
    env.path_taken = [origin]
    
    # Pre-compute distances to destination (needed for reward shaping in env)
    env._dist_to_dest = {}
    for node in env.location_names:
        try:
            env._dist_to_dest[node] = nx.shortest_path_length(env.graph, node, destination)
        except nx.NetworkXNoPath:
            env._dist_to_dest[node] = env.config.max_steps
            
    # Reset neighbors list based on new origin
    from src.utils.graph_utils import get_neighbors
    env._current_neighbors = get_neighbors(env.graph, origin)

    # Find nominal shortest path
    nominal_path = nx.shortest_path(env.graph, origin, destination, weight="base_time_hours")
    print(f"📍 Nominal Shortest Path: {' -> '.join(nominal_path)}")
    
    # Inject severe anomaly on the first edge of the nominal path
    if len(nominal_path) > 1:
        u, v = nominal_path[0], nominal_path[1]
        print(f"💥 Injecting SEVERE weather anomaly on edge {u} -> {v}")
        
        anom = ActiveAnomaly(
            anomaly_type="weather",
            severity=10.0,  # 10x travel time
            cost_multiplier=2.0,
            ticks_active=0
        )
        
        # Add to edge
        if (u, v) in env.anomaly_engine.edge_anomalies:
            env.anomaly_engine.edge_anomalies[(u, v)].append(anom)
        else:
            env.anomaly_engine.edge_anomalies[(u, v)] = [anom]
            
        # Also add to node v to make it very unattractive
        if v in env.anomaly_engine.node_anomalies:
            env.anomaly_engine.node_anomalies[v].append(anom)
        else:
            env.anomaly_engine.node_anomalies[v] = [anom]

    print("\n🚀 Agent Navigation Log:")
    
    log_lines = [
        "REROUTE TEST LOG",
        "================",
        f"Origin: {origin}",
        f"Destination: {destination}",
        f"Nominal Path: {' -> '.join(nominal_path)}",
        f"Injected Anomaly: {u} -> {v} (10x severity)",
        "----------------"
    ]
    
    done = False
    truncated = False
    
    while not (done or truncated):
        if agent is not None:
            state_dict = env.get_graph_state()
            state = feature_engine.build(state_dict).to(device)
            with torch.no_grad():
                action, _, _, _ = agent(state)
            a = action.cpu().numpy()
        else:
            a = env.action_space.sample()
            
        obs, reward, done, truncated, _ = env.step(a)
        
        leg = env.leg_details[-1]
        step_msg = f"Step {env.step_count}: {leg['from']} -> {leg['to']} | Vehicle: {leg['vehicle_type']} | Time: {leg['time_hours']:.1f}h"
        print(step_msg)
        log_lines.append(step_msg)

    print("\n🏁 Episode Complete")
    result_msg = f"Final Path: {' -> '.join(env.path_taken)}"
    print(result_msg)
    log_lines.append(result_msg)
    
    if (u, v) in zip(env.path_taken[:-1], env.path_taken[1:]):
        avoided = "❌ FAILED: Agent drove straight into the anomaly!"
    else:
        avoided = "✅ SUCCESS: Agent successfully rerouted around the anomaly!"
        
    print(avoided)
    log_lines.append(avoided)
    
    # Write log file
    with open("reroute_test.log", "w") as f:
        f.write("\n".join(log_lines))
    print("💾 Saved log to reroute_test.log")


if __name__ == "__main__":
    main()
