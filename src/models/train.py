"""
PPO Training Loop for the Graph-based Supply Chain Agent.

Handles rollout collection with HeteroData states, Advantage estimation (GAE),
and the PPO clipped surrogate objective update.

Outputs:
  - checkpoints/best_model.pt          Best model by avg reward
  - checkpoints/latest_model.pt        Latest model (saved every checkpoint_interval)
  - checkpoints/training_metrics.csv   Per-episode metrics
  - checkpoints/training_curves.png    Reward / delivery / loss plots
"""

import os
import sys
import csv
import time
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config.scenarios import india_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.ppo_agent import ActorCritic


# ═══════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ═══════════════════════════════════════════════════════════════════════

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

    def __len__(self):
        return len(self.states)


# ═══════════════════════════════════════════════════════════════════════
# GAE
# ═══════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    last_gae = 0

    vals = [v.item() for v in values] + [next_value.item()]

    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * vals[t + 1] * mask - vals[t]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages.insert(0, last_gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor([v.item() for v in values], dtype=torch.float32)

    return advantages, returns


# ═══════════════════════════════════════════════════════════════════════
# Checkpointing & Plotting
# ═══════════════════════════════════════════════════════════════════════

def save_checkpoint(agent, optimizer, episode, metrics, path, tag="latest"):
    """Save model + optimizer + training state."""
    torch.save({
        "episode": episode,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, os.path.join(path, f"{tag}_model.pt"))


def save_training_curves(metrics_history, path):
    """Generate and save training curve plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not found, skipping plot generation")
        return

    episodes = [m["episode"] for m in metrics_history]
    rewards = [m["reward"] for m in metrics_history]
    steps = [m["steps"] for m in metrics_history]
    delivered = [m["delivered"] for m in metrics_history]
    costs = [m["cost"] for m in metrics_history]
    times = [m["time_hours"] for m in metrics_history]

    # Compute rolling averages (window = 20)
    def rolling_avg(data, window=20):
        out = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            out.append(np.mean(data[start:i + 1]))
        return out

    reward_avg = rolling_avg(rewards)
    delivery_avg = rolling_avg([float(d) for d in delivered])
    steps_avg = rolling_avg(steps)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Supply Chain PPO — Training Curves", fontsize=16, fontweight="bold")
    fig.patch.set_facecolor("#0d1117")

    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # 1. Episode Reward
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="#58a6ff", linewidth=0.5)
    ax.plot(episodes, reward_avg, color="#58a6ff", linewidth=2, label="Rolling avg")
    ax.axhline(y=0, color="#f85149", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    # 2. Delivery Rate
    ax = axes[0, 1]
    ax.plot(episodes, [d * 100 for d in delivery_avg], color="#3fb950", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Delivery Rate (%)")
    ax.set_title("Delivery Success Rate (rolling)")
    ax.set_ylim(-5, 105)

    # 3. Episode Length
    ax = axes[0, 2]
    ax.plot(episodes, steps, alpha=0.3, color="#d2a8ff", linewidth=0.5)
    ax.plot(episodes, steps_avg, color="#d2a8ff", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")

    # 4. Total Cost
    ax = axes[1, 0]
    cost_avg = rolling_avg(costs)
    ax.plot(episodes, costs, alpha=0.3, color="#f0883e", linewidth=0.5)
    ax.plot(episodes, cost_avg, color="#f0883e", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("₹ Cost")
    ax.set_title("Episode Cost")

    # 5. Travel Time
    ax = axes[1, 1]
    time_avg = rolling_avg(times)
    ax.plot(episodes, times, alpha=0.3, color="#79c0ff", linewidth=0.5)
    ax.plot(episodes, time_avg, color="#79c0ff", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Hours")
    ax.set_title("Travel Time")

    # 6. Reward Distribution (last 100 episodes)
    ax = axes[1, 2]
    last_n = rewards[-min(100, len(rewards)):]
    ax.hist(last_n, bins=20, color="#58a6ff", alpha=0.7, edgecolor="#30363d")
    ax.axvline(x=np.mean(last_n), color="#f85149", linestyle="--", linewidth=2, label=f"Mean: {np.mean(last_n):.1f}")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.set_title(f"Reward Distribution (last {len(last_n)} eps)")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    plt.tight_layout()
    plt.savefig(os.path.join(path, "training_curves.png"), dpi=150, facecolor="#0d1117")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_ppo():
    """Main training loop with checkpointing and metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Output directory
    ckpt_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "checkpoints"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"🚀 Training PPO on device: {device}")
    print(f"📁 Checkpoints → {ckpt_dir}")

    # ── Setup ─────────────────────────────────────────────────
    config = india_scenario()
    env = SupplyChainEnv(config, render_mode=None)
    feature_engine = FeatureEngine()

    agent = ActorCritic(
        metadata=(['location', 'vehicle', 'shipment'],
                  [('location', 'route', 'location'),
                   ('vehicle', 'vehicle_at', 'location'),
                   ('shipment', 'shipment_at', 'location'),
                   ('shipment', 'shipment_dest', 'location'),
                   ('location', 'rev_vehicle_at', 'vehicle'),
                   ('location', 'rev_shipment_at', 'shipment'),
                   ('location', 'rev_shipment_dest', 'shipment')]),
        hidden_channels=64,
        out_channels=64,
        max_neighbors=env.max_neighbors,
        max_vehicles=env.max_vehicles
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)
    buffer = RolloutBuffer()

    # ── Hyperparameters (tuned) ───────────────────────────────
    num_episodes = 1000
    update_timestep = 256         # Larger batch size for more stable gradients
    ppo_epochs = 10               # More epochs per update for better gradient signal
    clip_param = 0.2
    entropy_coef_start = 0.05    # High entropy early → explore
    entropy_coef_end = 0.005     # Low entropy late → exploit
    vf_coef = 0.5
    checkpoint_interval = 50     # Save model every N episodes
    log_interval = 5             # Print every N episodes

    # ── Tracking ──────────────────────────────────────────────
    metrics_history = []         # Full history for CSV + plots
    recent_rewards = deque(maxlen=50)
    best_avg_reward = float("-inf")
    total_updates = 0
    time_step = 0
    start_time = time.time()

    # CSV header
    csv_path = os.path.join(ckpt_dir, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "steps", "delivered",
            "total_time_hours", "total_cost", "total_risk",
            "path", "shipment_type", "wall_clock_s"
        ])

    print(f"{'='*70}")
    print(f"  Episodes: {num_episodes} | Update every {update_timestep} steps")
    print(f"  Entropy: {entropy_coef_start} → {entropy_coef_end} (cosine decay)")
    print(f"  LR: 3e-4 → 1e-5 (cosine) | PPO epochs: {ppo_epochs}")
    print(f"  Checkpoint every {checkpoint_interval} eps | Log every {log_interval} eps")
    print(f"{'='*70}\n")

    # ── Training ──────────────────────────────────────────────
    for ep in range(1, num_episodes + 1):
        _, _ = env.reset(seed=ep)
        state_dict = env.get_graph_state()
        state = feature_engine.build(state_dict).to(device)

        ep_reward = 0.0
        done = False
        truncated = False

        while not (done or truncated):
            time_step += 1

            with torch.no_grad():
                action, log_prob, _, value = agent(state)

            _, reward, done, truncated, info = env.step(action.cpu().numpy())
            ep_reward += reward

            buffer.states.append(state.cpu())
            buffer.actions.append(action.cpu())
            buffer.log_probs.append(log_prob.cpu())
            buffer.rewards.append(reward)
            buffer.values.append(value.cpu())
            buffer.dones.append(done or truncated)

            if not (done or truncated):
                state_dict = env.get_graph_state()
                state = feature_engine.build(state_dict).to(device)

            # PPO Update
            if time_step % update_timestep == 0 and len(buffer) > 0:
                with torch.no_grad():
                    if done or truncated:
                        next_value = torch.tensor([0.0], device=device)
                    else:
                        _, _, _, next_value = agent(state)

                advantages, returns = compute_gae(
                    buffer.rewards, buffer.values, buffer.dones, next_value
                )

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.to(device)
                returns = returns.to(device)

                for _ in range(ppo_epochs):
                    actor_losses, critic_losses, entropies = [], [], []

                    for i in range(len(buffer)):
                        state_i = buffer.states[i].to(device)
                        action_i = buffer.actions[i].to(device).unsqueeze(0)
                        old_log_prob_i = buffer.log_probs[i].to(device).squeeze(-1)
                        adv_i = advantages[i]
                        ret_i = returns[i]

                        _, new_log_prob, entropy, new_value = agent(state_i, action=action_i)

                        ratio = torch.exp(new_log_prob - old_log_prob_i)
                        surr1 = ratio * adv_i
                        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_i
                        actor_losses.append(-torch.min(surr1, surr2))
                        critic_losses.append(nn.MSELoss()(new_value, ret_i.unsqueeze(0)))
                        entropies.append(entropy)

                    actor_loss = torch.cat(actor_losses).mean()
                    critic_loss = torch.stack(critic_losses).mean()
                    entropy_loss = -torch.cat(entropies).mean()

                    # Cosine-decay entropy coefficient
                    progress = min(ep / num_episodes, 1.0)
                    entropy_coef = entropy_coef_end + 0.5 * (entropy_coef_start - entropy_coef_end) * (1 + np.cos(np.pi * progress))

                    loss = actor_loss + vf_coef * critic_loss + entropy_coef * entropy_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

                total_updates += 1
                buffer.clear()
                time_step = 0

        # Step LR scheduler once per episode
        scheduler.step()

        # ── Record Metrics ─────────────────────────────────────
        delivered = env.current_node == env.destination
        elapsed = time.time() - start_time

        ep_metrics = {
            "episode": ep,
            "reward": round(ep_reward, 2),
            "steps": env.step_count,
            "delivered": delivered,
            "time_hours": round(env.total_time_hours, 1),
            "cost": round(env.total_cost, 0),
            "risk": round(env.total_risk, 3),
            "path": " → ".join(env.path_taken),
            "shipment_type": env.shipment.product_type,
            "wall_clock_s": round(elapsed, 1),
        }
        metrics_history.append(ep_metrics)
        recent_rewards.append(ep_reward)

        # Append to CSV (incremental so we never lose data)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep_metrics["episode"], ep_metrics["reward"], ep_metrics["steps"],
                ep_metrics["delivered"], ep_metrics["time_hours"], ep_metrics["cost"],
                ep_metrics["risk"], ep_metrics["path"], ep_metrics["shipment_type"],
                ep_metrics["wall_clock_s"],
            ])

        # ── Logging ────────────────────────────────────────────
        if ep % log_interval == 0:
            avg_r = np.mean(recent_rewards)
            recent_del = sum(1 for m in list(metrics_history)[-50:] if m["delivered"])
            del_rate = 100 * recent_del / min(50, ep)
            print(
                f"Ep {ep:>4d} │ R: {ep_reward:>8.1f} │ "
                f"Avg(50): {avg_r:>7.1f} │ "
                f"Del: {del_rate:>4.0f}% │ "
                f"Steps: {env.step_count:>3d} │ "
                f"Cost: ₹{env.total_cost:>10,.0f} │ "
                f"Updates: {total_updates} │ "
                f"⏱ {elapsed:.0f}s"
            )

        # ── Checkpointing ──────────────────────────────────────
        if ep % checkpoint_interval == 0:
            save_checkpoint(agent, optimizer, ep, metrics_history, ckpt_dir, tag="latest")
            save_training_curves(metrics_history, ckpt_dir)
            print(f"  💾 Saved latest checkpoint + training curves @ ep {ep}")

        # Best model
        if len(recent_rewards) >= 20:
            avg = np.mean(recent_rewards)
            if avg > best_avg_reward:
                best_avg_reward = avg
                save_checkpoint(agent, optimizer, ep, metrics_history, ckpt_dir, tag="best")
                print(f"  ⭐ New best model! Avg reward: {best_avg_reward:.2f} @ ep {ep}")

    # ── Final Save ────────────────────────────────────────────
    save_checkpoint(agent, optimizer, num_episodes, metrics_history, ckpt_dir, tag="final")
    save_training_curves(metrics_history, ckpt_dir)

    # Summary
    total_time = time.time() - start_time
    total_deliveries = sum(1 for m in metrics_history if m["delivered"])
    print(f"\n{'='*70}")
    print(f"  ✅ Training Complete!")
    print(f"     Total episodes:    {num_episodes}")
    print(f"     Total time:        {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"     Delivery rate:     {100*total_deliveries/num_episodes:.1f}%")
    print(f"     Best avg reward:   {best_avg_reward:.2f}")
    print(f"     Total PPO updates: {total_updates}")
    print(f"     Outputs saved to:  {ckpt_dir}/")
    print(f"       ├── best_model.pt")
    print(f"       ├── latest_model.pt")
    print(f"       ├── final_model.pt")
    print(f"       ├── training_metrics.csv")
    print(f"       └── training_curves.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    train_ppo()
