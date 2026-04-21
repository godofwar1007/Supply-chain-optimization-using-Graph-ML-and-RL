"""
PPO Training Loop for the Graph-based Supply Chain Agent.

Handles rollout collection with HeteroData states, Advantage estimation (GAE),
and the PPO clipped surrogate objective update.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.scenarios import india_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.ppo_agent import ActorCritic


class RolloutBuffer:
    """Stores transitions for PPO on graph data."""
    def __init__(self):
        self.states = []       # List of HeteroData
        self.actions = []      # List of tensors [2]
        self.log_probs = []    # List of tensors [1]
        self.rewards = []      # List of floats
        self.values = []       # List of tensors [1]
        self.dones = []        # List of bools
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    last_gae = 0
    
    # Append next_value to values list for easier indexing
    vals = [v.item() for v in values] + [next_value.item()]
    
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * vals[t + 1] * mask - vals[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages.insert(0, last_gae)
        
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor([v.item() for v in values], dtype=torch.float32)
    
    return advantages, returns


def train_ppo():
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training PPO on device: {device}")
    
    # Setup Env
    config = india_scenario()
    env = SupplyChainEnv(config, render_mode=None)
    feature_engine = FeatureEngine()
    
    # Setup Agent
    agent = ActorCritic(
        metadata=(['location', 'vehicle', 'shipment'], 
                  [('location', 'route', 'location'), 
                   ('vehicle', 'vehicle_at', 'location'), 
                   ('shipment', 'shipment_at', 'location'), 
                   ('shipment', 'shipment_dest', 'location'),
                   ('location', 'rev_route', 'location'),
                   ('location', 'rev_vehicle_at', 'vehicle'),
                   ('location', 'rev_shipment_at', 'shipment'),
                   ('location', 'rev_shipment_dest', 'shipment')]),
        hidden_channels=64,
        out_channels=64,
        max_neighbors=env.max_neighbors,
        max_vehicles=env.max_vehicles
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    buffer = RolloutBuffer()
    
    # Hyperparameters
    num_episodes = 500
    update_timestep = 100  # Update every 100 steps
    ppo_epochs = 4
    clip_param = 0.2
    entropy_coef = 0.01
    vf_coef = 0.5
    
    time_step = 0
    global_step = 0
    
    for ep in range(1, num_episodes + 1):
        _, _ = env.reset(seed=ep)
        state_dict = env.get_graph_state()
        state = feature_engine.build(state_dict).to(device)
        
        ep_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            time_step += 1
            global_step += 1
            
            # Select action
            with torch.no_grad():
                action, log_prob, _, value = agent(state)
                
            # Step environment
            _, reward, done, truncated, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            
            # Save to buffer
            buffer.states.append(state.cpu())
            buffer.actions.append(action.cpu())
            buffer.log_probs.append(log_prob.cpu())
            buffer.rewards.append(reward)
            buffer.values.append(value.cpu())
            buffer.dones.append(done or truncated)
            
            # Next state
            if not (done or truncated):
                state_dict = env.get_graph_state()
                state = feature_engine.build(state_dict).to(device)
                
            # Update PPO
            if time_step % update_timestep == 0:
                with torch.no_grad():
                    if done or truncated:
                        next_value = torch.tensor([0.0], device=device)
                    else:
                        _, _, _, next_value = agent(state)
                        
                advantages, returns = compute_gae(
                    buffer.rewards, buffer.values, buffer.dones, next_value
                )
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.to(device)
                returns = returns.to(device)
                
                for _ in range(ppo_epochs):
                    actor_losses, critic_losses, entropies = [], [], []
                    
                    for i in range(len(buffer.states)):
                        # Evaluate single action
                        state_i = buffer.states[i].to(device)
                        action_i = buffer.actions[i].to(device).unsqueeze(0)
                        old_log_prob_i = buffer.log_probs[i].to(device).squeeze(-1)
                        adv_i = advantages[i]
                        ret_i = returns[i]
                        
                        _, new_log_prob, entropy, new_value = agent(state_i, action=action_i)
                        
                        # Policy Loss
                        ratio = torch.exp(new_log_prob - old_log_prob_i)
                        surr1 = ratio * adv_i
                        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_i
                        actor_losses.append(-torch.min(surr1, surr2))
                        
                        # Value Loss
                        critic_losses.append(nn.MSELoss()(new_value, ret_i.unsqueeze(0)))
                        
                        # Entropy
                        entropies.append(entropy)
                        
                    # Aggregate losses
                    actor_loss = torch.cat(actor_losses).mean()
                    critic_loss = torch.stack(critic_losses).mean()
                    entropy_loss = -torch.cat(entropies).mean()
                    
                    # Total Loss
                    loss = actor_loss + vf_coef * critic_loss + entropy_coef * entropy_loss
                    
                    # Optimize
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                
                buffer.clear()
                time_step = 0
                
        if ep % 5 == 0:
            print(f"Episode {ep} \t Reward: {ep_reward:.2f} \t Steps: {env.step_count}")

if __name__ == "__main__":
    train_ppo()
