import torch
from torch import nn
from torch_geometric.nn import GATConv


class GNNImageClassificator(nn.Module):
    """
    See Figure 2 from https://iopscience.iop.org/article/10.1088/1742-6596/1871/1/012071/pdf
    """
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 152,
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.gat1 = GATConv(in_channels=self.in_channels, out_channels=self.hidden_dim)
        self.gat2 = GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        self.gat3 = GATConv(in_channels=self.in_channels + self.hidden_dim, out_channels=self.hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels + 3 * self.hidden_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes),
        )
    
    def tmp_forward(self, node_features, edge_indices) -> torch.Tensor:
        x1 = self.gat1(node_features, edge_indices)
        x2 = self.gat2(x1, edge_indices)
        
        tmp_x = torch.cat((node_features, x2), dim=-1)
        
        x3 = self.gat3(tmp_x, edge_indices)
        
        x4 = torch.cat((node_features, x1, x2, x3), dim=-1)
        
        return x4
    
    def forward(self, batch_node, batch_edge) -> torch.Tensor:
        batch_list = []
        
        for tmp1, tmp2 in zip(batch_node, batch_edge):
            batch_list.append(self.tmp_forward(node_features=tmp1, edge_indices=tmp2))
            
        new_features = torch.stack(batch_list, dim=0)
        new_features = new_features.mean(dim=1)
        
        logits = self.fc(new_features)
        
        return logits
        
        
        
        