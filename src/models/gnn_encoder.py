"""
Graph Neural Network Encoder for the heterogeneous supply chain graph.

Uses a Heterogeneous Graph Transformer (HGT) architecture to process
different node types (location, vehicle, shipment) and edge types (route,
vehicle_at, etc.) into unified embedding spaces.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear


class GNNEncoder(nn.Module):
    """
    Encodes a HeteroData object into node-level and graph-level embeddings.
    """

    def __init__(
        self,
        metadata: tuple,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """
        Parameters
        ----------
        metadata : tuple
            (node_types, edge_types) from hetero_data.metadata()
        hidden_channels : int
            Dimensionality of intermediate embeddings.
        out_channels : int
            Dimensionality of the final output embeddings.
        num_heads : int
            Number of attention heads in HGT.
        num_layers : int
            Number of message passing layers.
        """
        super().__init__()
        self.metadata = metadata
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # 1. Type-specific linear projections to bring all raw features 
        # to the same hidden dimensionality.
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            # Linear module from PyG automatically infers input size
            # on the first forward pass.
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # 2. HGT message passing layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)

        # 3. Final projection for the action heads
        self.out_lin = Linear(hidden_channels, out_channels)
        
        # 4. Global context processing
        # We know context dim is 10 + 2 progress features = 12, but we'll let Linear infer it
        self.context_mlp = nn.Sequential(
            Linear(-1, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, out_channels)
        )

    def forward(self, data: HeteroData) -> dict:
        """
        Forward pass.

        Returns
        -------
        dict
            Contains:
            - "node_embeddings": Dict[str, Tensor] of per-node-type embeddings
            - "graph_embedding": Tensor of shape [batch_size, out_channels]
        """
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 1. Project input features
        x_dict_proj = {}
        for node_type, x in x_dict.items():
            x_dict_proj[node_type] = self.lin_dict[node_type](x).relu()

        # 2. Message passing
        for conv in self.convs:
            x_dict_proj = conv(x_dict_proj, edge_index_dict)

        # 3. Final activation and projection
        out_dict = {}
        for node_type, x in x_dict_proj.items():
            out_dict[node_type] = self.out_lin(x).relu()

        # 4. Graph-level embedding (Mean pooling of Location + Shipment + Context)
        # Combine global context and step progress
        ctx = data.global_context  # Shape: [1, 10] (or [batch, 10])
        prog = data.step_progress  # Shape: [1, 2] (or [batch, 2])
        combined_ctx = torch.cat([ctx, prog], dim=-1)
        ctx_emb = self.context_mlp(combined_ctx)  # [batch, out_channels]

        # Mean pool locations and shipments to get graph structure summary
        # If batching is used in the future, we'd use global_mean_pool with batch index
        loc_emb = out_dict["location"].mean(dim=0, keepdim=True)
        ship_emb = out_dict["shipment"].mean(dim=0, keepdim=True)
        
        # Additive combination for global graph representation
        graph_embedding = ctx_emb + loc_emb + ship_emb

        return {
            "node_embeddings": out_dict,
            "graph_embedding": graph_embedding
        }
