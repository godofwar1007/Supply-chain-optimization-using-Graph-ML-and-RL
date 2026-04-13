import gymnasium as gym
import numpy as np
from environment import supply_chain_env_fixed_route
from agent import DQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm

def train():
    # Hyperparameters
    EPISODES = 500
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 100
    GAMMA = 0.99
    LEARNING_RATE = 1e-3
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    BUFFER_CAPACITY = 10000
    EVAL_INTERVAL = 50   # evaluate every N episodes
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
    
    print("Starting training...")
    
    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            agent.learn()   # learn after every step (or could learn every few steps)
            obs = next_obs
            total_reward += reward
        
        # Store episode stats
        episode_rewards.append(total_reward)
        episode_times.append(info['total_time'])
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_time = np.mean(episode_times[-10:])
            print(f"Episode {episode:4d} | Avg Reward (last 10): {avg_reward:.2f} | Avg Time: {avg_time:.2f} min | Epsilon: {agent.epsilon:.3f}")
        
        # Evaluate and save model
        if episode % EVAL_INTERVAL == 0:
            eval_reward = evaluate(env, agent, num_episodes=10)
            print(f"Evaluation after episode {episode}: Avg Reward = {eval_reward:.2f}")
            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(SAVE_PATH)
                print(f"  -> New best model saved with reward {best_reward:.2f}")
    
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

def plot_training(rewards, times):
    """Plot reward and travel time curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward (negative time)')
    ax1.set_title('Training Progress: Reward per Episode')
    
    ax2.plot(times)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Travel Time (min)')
    ax2.set_title('Training Progress: Total Time per Episode')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    trained_agent = train()
