import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class SpatialTemporal(nn.Module):
    # in_channels=32
    def __init__(self, in_channels, dilations, out_channels=None, dilation_channels=32, dropout=0.3):
        self.total_dilation = sum(dilations)
        self.blocks = len(dilations)

        self.gate1 = nn.ModuleList()
        self.gate2 = nn.ModuleList()


        for d in dilations:
            self.gate1(nn.Conv2d(in_channels=in_channels,
                                    out_channels=dilation_channels,
                                    kernel_size=(1, 2),dilation=d))
            
            self.gate2(nn.Conv2d(in_channels=in_channels,
                                    out_channels=dilation_channels,
                                    kernel_size=(1, 2),dilation=d))
            
            self.gcn_conv = GCNConv(in_channels=dilation_channels, out_channels=in_channels)
        
        if out_channels != None:
            self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        
    def forward(self, input, edge_index, edge_weight=None):
        # (batch_size, residual_channels, num_nodes, t)
        if self.total_dilation > input.shape[-2]:
            input = F.pad(input, (0, 0, self.total_dilation - input.shape[-2] + 1, 0))

        g1 = input
        g2 = input

        for i in range(len(self.blocks)):
            g1 = self.gate1[i](g1)
            g2 = self.gate2[i](g2)

            g1 = F.tanh(g1).transpose(-3, -2)
            g2 = F.sigmoid(g2).transpose(-3, -2)

            # (batch_size, num_nodes,  in_channels, t_cur - d[i])

            x = self.gcn_conv(g1 * g2, edge_index, edge_weight).transpose(-3, -2)

            # (batch_size, in_channels, num_nodes, t_cur - d[i])

            g1 = x
            g2 = x

        # (batch_size, in_channels, num_nodes, t - total_dilation)
        x = input[..., -1] + x[..., -1]
        # (batch_size, in_channels, num_nodes, 1)


        if self.out != None:
            x = self.out(x)

        # (batch_size, out_channels, num_nodes, 1)  
        return x
        
        


    
class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, adj, in_channels, out_channels, out_timestamps, dilations=[1,2,1,2,1,2,1,2], dropout=0.3, residual_channels=32, dilation_channels=32, skip_channels=256, layers=2):
        self.input = nn.Conv2d(in_features=in_channels, out_features=residual_channels, kernel_size=(1, 1))

        self.spatial_temporals = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(layers):
            self.spatial_temporals.append(SpatialTemporal(residual_channels, dilations, out_channels=skip_channels, dilation_channels=dilation_channels, dropout=dropout))
            self.bns.append(nn.BatchNorm2d(skip_channels))

        self.end1 = nn.Conv2d(in_features=in_channels, out_features=out_channels, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(in_features=in_channels, out_features=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        # (batch_size, t, num_nodes, input_dim)

        x = x.transpose(1, 3)

        # (batch_size, input_dim, num_nodes, t)
        x = self.input(x)

        # (batch_size, residual_channels, num_nodes, t)
        for k in len(self.spatial_temporals):
            x = self.spatial_temporals[k](x)



    
