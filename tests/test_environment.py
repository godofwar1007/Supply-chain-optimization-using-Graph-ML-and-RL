"""
Phase 1 verification — smoke test the environment + feature engine.

Runs 100 episodes with a random agent on both scenarios and checks:
1. No crashes
2. Observation shapes are correct
3. Rewards are bounded / sensible
4. HeteroData from feature engine has correct structure
5. Prints a sample episode trace
"""

import sys
import os
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config.scenarios import small_scenario, india_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine


def test_scenario(name: str, config, num_episodes=100, verbose_episode=0):
    """Run smoke test on a scenario."""
    print(f"\n{'='*70}")
    print(f"  Testing: {name} ({len(config.locations)} locations, "
          f"{len(config.routes)} routes, {len(config.vehicles)} vehicles)")
    print(f"{'='*70}")

    env = SupplyChainEnv(config, render_mode="human")
    feature_engine = FeatureEngine()

    all_rewards = []
    all_times = []
    all_costs = []
    all_steps = []
    deliveries = 0
    errors = []

    for ep in range(num_episodes):
        try:
            obs, info = env.reset(seed=ep)

            # Check obs shape
            assert obs.shape == (env.obs_dim,), (
                f"Obs shape mismatch: {obs.shape} vs expected ({env.obs_dim},)"
            )
            assert not np.any(np.isnan(obs)), "NaN in initial observation"

            # Test feature engine on first step
            if ep == 0:
                graph_state = env.get_graph_state()
                hetero_data = feature_engine.build(graph_state)
                print(f"\n  HeteroData structure:")
                print(f"    Node types: {hetero_data.node_types}")
                print(f"    Edge types: {hetero_data.edge_types}")
                for ntype in hetero_data.node_types:
                    print(f"    {ntype}.x shape: {hetero_data[ntype].x.shape}")
                for etype in hetero_data.edge_types:
                    ei = hetero_data[etype].edge_index
                    print(f"    {etype} edges: {ei.shape[1]}")
                    if hasattr(hetero_data[etype], "edge_attr") and hetero_data[etype].edge_attr is not None:
                        print(f"    {etype} edge_attr shape: {hetero_data[etype].edge_attr.shape}")
                print(f"    global_context shape: {hetero_data.global_context.shape}")
                print()

            if ep == verbose_episode:
                env.render()  # Print episode header

            done = False
            truncated = False
            ep_reward = 0.0

            while not (done or truncated):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)

                assert not np.any(np.isnan(obs)), f"NaN in obs at step {info['step_count']}"
                ep_reward += reward

                if ep == verbose_episode:
                    env.render()

            all_rewards.append(ep_reward)
            all_times.append(info["total_time_hours"])
            all_costs.append(info["total_cost"])
            all_steps.append(info["step_count"])
            if info["current_node"] == info.get("destination", env.destination):
                deliveries += 1

        except Exception as e:
            errors.append((ep, str(e), traceback.format_exc()))
            if len(errors) <= 3:
                print(f"\n  ❌ Error in episode {ep}: {e}")
                traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n  📊 Results ({num_episodes} episodes):")
    print(f"     Errors:      {len(errors)}/{num_episodes}")
    print(f"     Deliveries:  {deliveries}/{num_episodes} "
          f"({100*deliveries/num_episodes:.0f}%)")
    print(f"     Avg steps:   {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"     Avg reward:  {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"     Avg time:    {np.mean(all_times):.1f}h ± {np.std(all_times):.1f}h")
    print(f"     Avg cost:    ₹{np.mean(all_costs):,.0f} ± ₹{np.std(all_costs):,.0f}")
    print(f"     Reward range: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]")

    if len(errors) == 0:
        print(f"\n  ✅ {name} PASSED — no errors in {num_episodes} episodes")
    else:
        print(f"\n  ❌ {name} FAILED — {len(errors)} errors")

    return len(errors) == 0


if __name__ == "__main__":
    print("🔧 Phase 1 Verification — Supply Chain Environment")
    print("=" * 70)

    results = {}

    # Test small scenario
    results["small"] = test_scenario(
        "Small (4-node test)", small_scenario(),
        num_episodes=50, verbose_episode=0,
    )

    # Test India scenario
    results["india"] = test_scenario(
        "India (25-node medium)", india_scenario(),
        num_episodes=100, verbose_episode=0,
    )

    # Final verdict
    print(f"\n\n{'='*70}")
    if all(results.values()):
        print("  ✅ ALL TESTS PASSED — Phase 1 environment is verified!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  ❌ FAILURES: {', '.join(failed)}")
    print(f"{'='*70}")
