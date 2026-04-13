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
        print(f"\n{'='*60}")
        print(f"EPISODE {ep+1}")
        print(f"{'='*60}")
        
        obs, _ = env.reset()
        done = False
        total_time = 0.0
        
        while not done:
            if render:
                env.render()
                
            action = agent.act(obs, eval_mode=True)
            next_obs, reward, done, _, info = env.step(action)
            
            travel_time = -reward
            total_time += travel_time
            
            print(f"  \033[94mSelected Edge {action}\033[0m | Segment Travel: {travel_time:.2f} min")
            
            obs = next_obs
        
        env.render() # Final render to show completion
        print(f"\033[1mActions Taken Summary:\033[0m (last episode results above)")

def run_multiple_seeds(agent, env, num_seeds=10):
    """
    Test agent on multiple random seeds (different anomaly evolutions).
    """
    total_times = []
    print(f"\n{'='*60}")
    print(f"{'Seed':<10} | {'Total Time (min)':<20}")
    print("-" * 35)
    for seed in range(num_seeds):
        obs, _ = env.reset(seed=seed)
        done = False
        total_time = 0.0
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, reward, done, _, info = env.step(action)
            total_time += -reward
        total_times.append(total_time)
        print(f"{seed:<10} | {total_time:<20.2f}")
    
    print("-" * 35)
    print(f"\033[1;92mAverage Over {num_seeds} Seeds: {np.mean(total_times):.2f} \u00b1 {np.std(total_times):.2f} min\033[0m")

if __name__ == "__main__":
    # Load environment and agent
    env = supply_chain_env_fixed_route()
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = env.action_space.n
    
    agent = DQNAgent(obs_dim, action_dim)
    try:
        agent.load("dqn_model.pth")
        print("\033[92mModel loaded successfully.\033[0m")
    except Exception as e:
        print(f"\033[91mCould not load model: {e}\033[0m")
        print("Please train the model first using train.py")
        exit(1)
    
    visualize(agent, env, num_episodes=3, render=True)
    run_multiple_seeds(agent, env, num_seeds=10)
