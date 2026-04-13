import gymnasium as gym
import numpy as np
from environment import supply_chain_env_fixed_route
from agent import DQNAgent
import matplotlib.pyplot as plt

def visualize(agent, env, num_episodes=5, render=True):
    """
    Run episodes with the trained agent and print detailed results.
    """
    for ep in range(num_episodes):
        print(f"\n{'='*50}")
        print(f"Episode {ep+1}")
        print(f"{'='*50}")
        
        obs, _ = env.reset()
        done = False
        step = 0
        total_time = 0.0
        actions_taken = []
        anomalies_per_step = []
        
        while not done:
            # Get current segment info BEFORE taking step (since step_idx is still valid)
            src, tgt = env.segments[step]
            edges = env._get_current_segment_edges()
            
            action = agent.act(obs, eval_mode=True)
            next_obs, reward, done, _, info = env.step(action)
            
            chosen_edge = next(e for e in edges if e["edge_id"] == action)
            
            # Anomaly factors for this segment's edges
            anomaly_factors = {}
            for e in edges:
                factor = env.anomaly_factor(e["source"], e["target"], e["edge_id"])
                anomaly_factors[e["edge_id"]] = factor
            
            travel_time = -reward  # reward is negative travel time
            total_time += travel_time
            
            print(f"\nStep {step+1}: {src} → {tgt}")
            print(f"  Chosen edge ID: {action}")
            print(f"  Chosen edge base time: {chosen_edge['base_time_min']} min")
            print(f"  Anomaly factors: {anomaly_factors}")
            print(f"  Travel time: {travel_time:.2f} min")
            
            actions_taken.append(action)
            anomalies_per_step.append(anomaly_factors)
            
            if render:
                env.render()  # uses simple print render
            obs = next_obs
            step += 1
        
        print(f"\nTotal travel time: {total_time:.2f} min")
        print(f"Actions taken: {actions_taken}")

def run_multiple_seeds(agent, env, num_seeds=10):
    """
    Test agent on multiple random seeds (different anomaly evolutions).
    """
    total_times = []
    for seed in range(num_seeds):
        obs, _ = env.reset(seed=seed)
        done = False
        total_time = 0.0
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, reward, done, _, info = env.step(action)
            total_time += -reward  # accumulate travel time
        total_times.append(total_time)
        print(f"Seed {seed:2d}: total time = {total_time:.2f} min")
    
    print(f"\nAverage over {num_seeds} seeds: {np.mean(total_times):.2f} ± {np.std(total_times):.2f} min")

if __name__ == "__main__":
    # Load environment and agent
    env = supply_chain_env_fixed_route()
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = env.action_space.n
    
    agent = DQNAgent(obs_dim, action_dim)
    agent.load("dqn_model.pth")   # make sure the path is correct
    
    # Option 1: Visualize a few episodes with rendering
    visualize(agent, env, num_episodes=3, render=True)
    
    # Option 2: Test robustness over different random seeds
    print("\n" + "="*50)
    print("Testing over different random seeds (anomaly variations)")
    print("="*50)
    run_multiple_seeds(agent, env, num_seeds=10)
