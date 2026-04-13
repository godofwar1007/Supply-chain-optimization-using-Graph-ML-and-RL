import gymnasium as gym
import numpy as np
from environment import supply_chain_env_fixed_route
from agent import DQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm

def train():
    # Hyperparameters for higher stability and better peak performance
    EPISODES = 1500
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 200 # increased for stability
    GAMMA = 0.99
    LEARNING_RATE = 5e-4 # lowered to prevent divergence
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.997 # slower decay for better exploration
    BUFFER_CAPACITY = 20000
    EVAL_INTERVAL = 50
    SAVE_PATH = "dqn_model.pth"

    # Create environment
    env = supply_chain_env_fixed_route()
    
    # Get observation and action dimensions
    obs, _ = env.reset()
    obs_dim = len(obs)
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )
    
    # Tracking
    episode_rewards = []
    episode_times = []
    best_reward = -np.inf
    
    print("Starting optimized training...")
    
    pbar = tqdm(range(1, EPISODES + 1))
    for episode in pbar:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            # Normalize reward for stability
            agent.remember(obs, action, reward / 100.0, next_obs, done)
            agent.learn()
            obs = next_obs
            total_reward += reward
        
        episode_rewards.append(total_reward)
        episode_times.append(info['total_time'])
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_time = np.mean(episode_times[-10:])
            pbar.set_postfix({
                "Avg Reward": f"{avg_reward:.1f}",
                "Best": f"{best_reward:.1f}",
                "Epsilon": f"{agent.epsilon:.3f}"
            })
        
        if episode % EVAL_INTERVAL == 0:
            eval_reward = evaluate(env, agent, num_episodes=15) # more episodes for more accurate eval
            tqdm.write(f"Episode {episode}: Eval Reward = {eval_reward:.2f}")
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(SAVE_PATH)
                tqdm.write(f"  \033[92m-> New best model saved (Reward: {best_reward:.2f})\033[0m")
    
    env.close()
    
    # Plot results
    plot_training(episode_rewards, episode_times)
    
    return agent

def evaluate(env, agent, num_episodes=10):
    """Evaluate agent without exploration."""
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.act(obs, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    return np.mean(total_rewards)

def plot_training(rewards, times, window=50):
    """Plot reward and travel time curves with moving averages."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Helper for moving average
    def moving_average(data, w):
        if len(data) < w: return np.full(len(data), np.nan)
        return np.convolve(data, np.ones(w), 'valid') / w

    # Plot Rewards
    ax1.scatter(range(len(rewards)), rewards, alpha=0.1, color='blue', s=2, label='Raw Episodes')
    ma_rewards = moving_average(rewards, window)
    ax1.plot(np.arange(window-1, len(rewards)), ma_rewards[~np.isnan(ma_rewards)], 
             color='red', linewidth=2, label=f'{window}-Ep Moving Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (negative time)')
    ax1.set_title('Training Progress: Reward per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Times
    ax2.scatter(range(len(times)), times, alpha=0.1, color='green', s=2, label='Raw Episodes')
    ma_times = moving_average(times, window)
    ax2.plot(np.arange(window-1, len(times)), ma_times[~np.isnan(ma_times)], 
             color='darkgreen', linewidth=2, label=f'{window}-Ep Moving Avg')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Travel Time (min)')
    ax2.set_title('Training Progress: Total Time per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    trained_agent = train()
