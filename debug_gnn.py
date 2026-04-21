import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

# Mock data
data = HeteroData()
data['location'].x = torch.randn(4, 16)
data['vehicle'].x = torch.randn(2, 16)
data['shipment'].x = torch.randn(1, 16)

data['location', 'route', 'location'].edge_index = torch.tensor([[0, 1], [1, 2]])
data['vehicle', 'vehicle_at', 'location'].edge_index = torch.tensor([[0, 1], [0, 2]])
data['shipment', 'shipment_at', 'location'].edge_index = torch.tensor([[0], [0]])

conv = HGTConv(16, 16, data.metadata(), heads=2)
out = conv(data.x_dict, data.edge_index_dict)
print("Keys in out:", out.keys())
