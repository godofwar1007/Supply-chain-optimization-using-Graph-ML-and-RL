import gymnasium as gym
import numpy as np
from environment import supply_chain_env_fixed_route
from agent import DQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
import os

def train(trial=None):
    # Default Hyperparameters (Absolute Best from Optuna Study)
    EPISODES = 500 
    BATCH_SIZE = 128
    TARGET_UPDATE_FREQ = 400
    GAMMA = 0.938
    LEARNING_RATE = 5.16e-4
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.994 # Reverted to best decay
    BUFFER_CAPACITY = 20000
    EVAL_INTERVAL = 50
    SAVE_PATH = "dqn_model.pth"
    PATIENCE = 3 # Stop if no improvement for 150 episodes

    if trial:
        # Optuna suggested hyperparameters
        LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        BATCH_SIZE = trial.suggest_categorical("batch_size", [32, 64, 128])
        GAMMA = trial.suggest_float("gamma", 0.9, 0.999)
        EPSILON_DECAY = trial.suggest_float("epsilon_decay", 0.98, 0.995)
        TARGET_UPDATE_FREQ = trial.suggest_int("target_update_freq", 100, 500, step=50)
        # We can also tune the episodes if needed, but usually kept constant for comparison
        EPISODES = 400 # Shorter episodes for faster tuning

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
    no_improvement_count = 0
    
    print(f"Starting {'tuning' if trial else 'optimized'} training...")
    
    pbar = tqdm(range(1, EPISODES + 1), disable=(trial is not None))
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
        
        if episode % 10 == 0 and not trial:
            avg_reward = np.mean(episode_rewards[-10:])
            pbar.set_postfix({
                "Avg Reward": f"{avg_reward:.1f}",
                "Best": f"{best_reward:.1f}",
                "Epsilon": f"{agent.epsilon:.3f}"
            })
        
        if episode % EVAL_INTERVAL == 0:
            eval_reward = evaluate(env, agent, num_episodes=10)
            if not trial:
                tqdm.write(f"Episode {episode}: Eval Reward = {eval_reward:.2f}")
            
            if eval_reward > best_reward:
                best_reward = eval_reward
                no_improvement_count = 0
                if not trial: # Only save best model during final run
                    agent.save(SAVE_PATH)
            else:
                no_improvement_count += 1
            
            # Optuna Pruning
            if trial:
                trial.report(eval_reward, episode)
                if trial.should_prune():
                    env.close()
                    raise optuna.exceptions.TrialPruned()
            
            # Early Stopping
            if no_improvement_count >= PATIENCE:
                if not trial:
                    tqdm.write(f"\nEarly stopping at episode {episode} due to no improvement.")
                break
    
    env.close()
    
    if not trial:
        # Plot results for final run
        plot_training(episode_rewards, episode_times)
        return agent
    
    return best_reward # Return best evaluation reward for Optuna

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--trials", type=int, default=20, help="Number of tuning trials")
    args = parser.parse_args()

    if args.tune:
        print(f"Starting Optuna study with {args.trials} trials...")
        study = optuna.create_study(direction="maximize")
        study.optimize(train, n_trials=args.trials)
        
        print("\nStudy finished!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"  Value: {study.best_trial.value}")
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        # Standard training with defaults
        trained_agent = train()
