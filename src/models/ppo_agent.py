"""
PPO Agent with Graph Attention Actor-Critic.

Uses the GNNEncoder to process HeteroData. The Actor predicts action
distributions for neighbor selection and vehicle selection using attention.
The Critic predicts the value function using the global graph embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.models.gnn_encoder import GNNEncoder


class ActorCritic(nn.Module):
    """
    Combines the GNN Encoder with Actor and Critic heads.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_neighbors: int = 20,
        max_vehicles: int = 50,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            metadata=metadata,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        
        self.max_neighbors = max_neighbors
        self.max_vehicles = max_vehicles

        # ── Critic Head ───────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

        # ── Actor Heads (Pointer Networks) ─────────────────────────────
        # Queries depend on the graph embedding + current node embedding
        self.query_proj_node = nn.Linear(out_channels * 2, out_channels)
        self.query_proj_veh = nn.Linear(out_channels * 2, out_channels)
        
        # Keys project the candidate nodes
        self.key_proj_node = nn.Linear(out_channels, out_channels)
        self.key_proj_veh = nn.Linear(out_channels, out_channels)

    def forward(self, data: HeteroData, action=None):
        """
        Forward pass for action selection and value prediction.
        
        Parameters
        ----------
        data : HeteroData
            The graph state. Must contain `neighbor_indices` and `current_node_idx`.
        action : Optional[Tensor]
            If provided, evaluates the log probability of this action.
            
        Returns
        -------
        If action is None:
            action, log_prob, entropy, value
        If action provided:
            log_prob, entropy, value
        """
        # 1. Encode graph
        enc_out = self.encoder(data)
        node_emb = enc_out["node_embeddings"]
        graph_emb = enc_out["graph_embedding"]  # [batch, out_channels]
        
        # We only support batch_size=1 for now given HeteroData handling without PyG DataLoaders
        # To support batched HeteroData properly, we'd use batch indices.
        batch_size = graph_emb.size(0)
        
        # 2. Critic Value
        value = self.critic(graph_emb).squeeze(-1)  # [batch]
        
        # 3. Actor - Extract embeddings for queries
        loc_emb = node_emb["location"]
        veh_emb = node_emb["vehicle"]
        
        # Current node embedding
        curr_idx = getattr(data, "current_node_idx", torch.tensor([0], device=graph_emb.device))
        curr_node_emb = loc_emb[curr_idx]  # [1, out_channels]
        
        # Query features: [graph_emb, curr_node_emb]
        query_feat = torch.cat([graph_emb, curr_node_emb], dim=-1)  # [1, out_channels * 2]
        
        q_node = self.query_proj_node(query_feat)  # [1, out_channels]
        q_veh = self.query_proj_veh(query_feat)    # [1, out_channels]
        
        # 4. Actor - Neighbor Selection Logits
        neighbor_indices = getattr(data, "neighbor_indices", torch.tensor([0], device=graph_emb.device))
        num_neighbors = neighbor_indices.size(0)
        
        if num_neighbors > 0:
            k_node = self.key_proj_node(loc_emb[neighbor_indices])  # [num_neighbors, out_channels]
            # Dot product attention
            node_scores = torch.matmul(q_node, k_node.T).squeeze(0)  # [num_neighbors]
        else:
            node_scores = torch.zeros(0, device=graph_emb.device)
            
        # Pad neighbor logits to fixed action space (max_neighbors)
        padded_node_logits = torch.full((self.max_neighbors,), float('-inf'), device=graph_emb.device)
        valid_n = min(num_neighbors, self.max_neighbors)
        if valid_n > 0:
            padded_node_logits[:valid_n] = node_scores[:valid_n]
        else:
            padded_node_logits[0] = 0.0  # Fallback to prevent NaN
            
        # 5. Actor - Vehicle Selection Logits
        num_vehicles = veh_emb.size(0)
        if num_vehicles > 0:
            k_veh = self.key_proj_veh(veh_emb)  # [num_vehicles, out_channels]
            veh_scores = torch.matmul(q_veh, k_veh.T).squeeze(0)  # [num_vehicles]
        else:
            veh_scores = torch.zeros(0, device=graph_emb.device)
            
        padded_veh_logits = torch.full((self.max_vehicles,), float('-inf'), device=graph_emb.device)
        valid_v = min(num_vehicles, self.max_vehicles)
        if valid_v > 0:
            padded_veh_logits[:valid_v] = veh_scores[:valid_v]
        else:
            padded_veh_logits[0] = 0.0
            
        # 6. Create Distributions
        dist_node = Categorical(logits=padded_node_logits.unsqueeze(0))
        dist_veh = Categorical(logits=padded_veh_logits.unsqueeze(0))
        
        if action is None:
            a_node = dist_node.sample()
            a_veh = dist_veh.sample()
            action = torch.stack([a_node, a_veh], dim=-1)  # [batch, 2]
            
        # Calculate log prob and entropy
        a_node, a_veh = action[:, 0], action[:, 1]
        log_prob = dist_node.log_prob(a_node) + dist_veh.log_prob(a_veh)
        entropy = dist_node.entropy() + dist_veh.entropy()
        
        if action.dim() == 2 and action.size(0) == 1:
            action = action.squeeze(0)  # Return flat array for gym
            
        return action, log_prob, entropy, value
