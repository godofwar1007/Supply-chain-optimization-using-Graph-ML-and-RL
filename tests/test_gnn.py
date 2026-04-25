"""
Unit test for the GNN Encoder.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.config.scenarios import small_scenario
from src.environment.supply_chain_env import SupplyChainEnv
from src.features.feature_engine import FeatureEngine
from src.models.gnn_encoder import GNNEncoder

def test_gnn_forward_pass():
    print("🔧 Testing GNN Encoder Forward Pass...")
    
    # 1. Generate sample HeteroData from environment
    env = SupplyChainEnv(small_scenario(), render_mode=None)
    env.reset(seed=42)
    state = env.get_graph_state()
    
    feature_engine = FeatureEngine()
    hetero_data = feature_engine.build(state)
    
    # 2. Initialize Encoder
    # Using small dimensionality for testing
    encoder = GNNEncoder(
        metadata=hetero_data.metadata(),
        hidden_channels=32,
        out_channels=16,
        num_heads=2,
        num_layers=2
    )
    
    # 3. Forward Pass
    try:
        out = encoder(hetero_data)
        print("✅ Forward pass successful.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise e
        
    # 4. Verify output structure and shapes
    node_emb = out["node_embeddings"]
    graph_emb = out["graph_embedding"]
    
    assert "location" in node_emb
    assert "vehicle" in node_emb
    assert "shipment" in node_emb
    
    num_locs = hetero_data["location"].x.shape[0]
    num_vehs = hetero_data["vehicle"].x.shape[0]
    num_ships = hetero_data["shipment"].x.shape[0]
    
    assert node_emb["location"].shape == (num_locs, 16), f"Expected ({num_locs}, 16), got {node_emb['location'].shape}"
    assert node_emb["vehicle"].shape == (num_vehs, 16)
    assert node_emb["shipment"].shape == (num_ships, 16)
    assert graph_emb.shape == (1, 16), f"Expected (1, 16), got {graph_emb.shape}"
    
    print("✅ Output shapes verified:")
    print(f"  - Location: {node_emb['location'].shape}")
    print(f"  - Vehicle:  {node_emb['vehicle'].shape}")
    print(f"  - Shipment: {node_emb['shipment'].shape}")
    print(f"  - Graph:    {graph_emb.shape}")
    
if __name__ == "__main__":
    test_gnn_forward_pass()
