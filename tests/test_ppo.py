"""
Unit test for the PPO Agent (Actor-Critic).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config.scenarios import small_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.ppo_agent import ActorCritic

def test_ppo_forward_pass():
    print("🔧 Testing PPO Agent Forward Pass...")
    
    # 1. Setup env and feature engine
    env = SupplyChainEnv(small_scenario(), render_mode=None)
    obs, _ = env.reset(seed=42)
    state = env.get_graph_state()
    
    feature_engine = FeatureEngine()
    hetero_data = feature_engine.build(state)
    
    # 2. Initialize Agent
    agent = ActorCritic(
        metadata=hetero_data.metadata(),
        hidden_channels=32,
        out_channels=16,
        num_heads=2,
        num_layers=2,
        max_neighbors=env.max_neighbors,
        max_vehicles=env.max_vehicles
    )
    
    # 3. Action Selection (No action provided)
    try:
        action, log_prob, entropy, value = agent(hetero_data)
        print("✅ Action selection successful.")
        
        # Check shapes
        assert action.shape == (2,), f"Expected action shape (2,), got {action.shape}"
        assert log_prob.shape == (1,), f"Expected log_prob shape (1,), got {log_prob.shape}"
        assert entropy.shape == (1,), f"Expected entropy shape (1,), got {entropy.shape}"
        assert value.shape == (1,), f"Expected value shape (1,), got {value.shape}"
        
        # Ensure selected actions are within bounds
        a_node, a_veh = action[0].item(), action[1].item()
        print(f"  Selected neighbor idx: {a_node}")
        print(f"  Selected vehicle idx:  {a_veh}")
        assert a_node < env.max_neighbors
        assert a_veh < env.max_vehicles
        
    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        raise e
        
    # 4. Action Evaluation (Action provided)
    try:
        # Provide the same action to evaluate log_prob
        # Add batch dim for evaluation (simulating what PPO update does)
        action_batch = action.unsqueeze(0)
        _, log_prob_eval, entropy_eval, value_eval = agent(hetero_data, action=action_batch)
        print("✅ Action evaluation successful.")
        
        assert log_prob_eval.shape == (1,)
        # Should match exactly since network is deterministic and state is the same
        assert torch.isclose(log_prob, log_prob_eval, atol=1e-5).all()
        
    except Exception as e:
        print(f"❌ Action evaluation failed: {e}")
        raise e

    print("✅ All PPO Agent tests passed.")
    
if __name__ == "__main__":
    test_ppo_forward_pass()
