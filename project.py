from functools import reduce
from operator import mul
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

class SpatialTemporal(nn.Module):
    def __init__(self, in_channels, dilation, out_channels=None, dilation_channels=32, dropout=0.3):
        super(SpatialTemporal, self).__init__()

        self.dropout = dropout

        self.gate1 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=dilation_channels,
                                    kernel_size=(1, 2),dilation=dilation)
            
        self.gate2 = nn.Conv2d(in_channels=in_channels,
                                    out_channels=dilation_channels,
                                    kernel_size=(1, 2),dilation=dilation)
            
        self.gcn_conv = GCNConv(in_channels=dilation_channels, out_channels=in_channels)
        
        if out_channels != None:
            self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        else:
            self.out = None

        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, input, edge_index, edge_weight=None):
        # (batch_size, t, num_nodes, in_channels)
        input = input.transpose(-3, -1)

        g1 = self.gate1(input)
        g2 = self.gate2(input)

        # (batch_size, dilation_channels, num_nodes, t_cur - d[i])

        g1 = F.tanh(g1)
        g1 = g1.transpose(-3, -1)
        g2 = F.sigmoid(g2)
        g2 = g2.transpose(-3, -1)

        # (batch_size, t_cur - d[i], num_nodes, dilation_channels)

        gcn_out = None
        cur_shape = list(g1.shape)
        batch_size = reduce(mul, cur_shape[:-2])
        g1 = torch.reshape(g1, (batch_size, cur_shape[-2], cur_shape[-1]))
        g2 = torch.reshape(g2, (batch_size, cur_shape[-2], cur_shape[-1]))

        for batch in range(len(g1)): 
            g = self.gcn_conv(g1[batch] * g2[batch], edge_index, edge_weight)
            if gcn_out == None:
                gcn_out = torch.unsqueeze(g, 0)
            else:
                gcn_out = torch.cat([gcn_out, torch.unsqueeze(g, 0)])

        gcn_out = torch.reshape(gcn_out, tuple(cur_shape[:-1] + [self.gcn_conv.out_channels]))


        # (batch_size, t_cur - d[i], num_nodes, in_channels)
        gcn_out = gcn_out.transpose(-3, -1)

        if self.out != None:
            gcn_out = self.out(gcn_out)

        # (batch_size, out_channels, num_nodes, t-dilation)
        gcn_out = gcn_out.transpose(-3, -1)  
        return gcn_out
        
        


    
class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, adj, in_channels, out_channels, out_timestamps, dilations=[1,2,1,2,1,2,1,2], dropout=0.3, residual_channels=32, dilation_channels=32, skip_channels=256):
        super(GraphWaveNet, self).__init__()

        self.total_dilation = sum(dilations)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=residual_channels, kernel_size=(1, 1))

        self.adj = adj
        self.num_nodes = num_nodes

        self.e1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.e2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)

        self.spatial_temporals = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip = nn.ModuleList()
        
        for d in dilations:
            self.spatial_temporals.append(SpatialTemporal(residual_channels, d, dilation_channels=dilation_channels, dropout=dropout))
            self.skip.append(nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1)))
            self.bns.append(nn.BatchNorm2d(residual_channels))

        self.end_tmp = nn.Conv2d(in_channels=skip_channels, out_channels=512, kernel_size=(1, 1))
        self.end1 = nn.Conv2d(in_channels=512, out_channels=out_timestamps, kernel_size=(1, 1))
        self.end2 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        # (batch_size, t, num_nodes, input_dim)

        edge_list = [[], []]
        edge_weight = []

        adp = F.softmax(F.relu(torch.mm(self.e1, self.e2)), dim=1)
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj.item((i, j)) != 0:
                    edge_list[0].append(i)
                    edge_list[1].append(j)
                    edge_weight.append(self.adj.item((i, j)) + adp[i][j])

        edge_list = torch.tensor(edge_list)
        edge_weight = torch.tensor(edge_weight)
        x = x.transpose(-3, -1)

        # (batch_size, input_dim, num_nodes, t)
        x = self.input(x)
        if self.total_dilation > x.shape[-1]:
            x = F.pad(x, (self.total_dilation - x.shape[-1] + 1, 0))

        skip_out = None

        # (batch_size, residual_channels, num_nodes, t)
        for k in range(len(self.spatial_temporals)):
            x = x.transpose(-3, -1)
            # (batch_size, t_cur, num_nodes, residual_channels)
            x = self.spatial_temporals[k](x, edge_list, edge_weight)
            # (batch_size, t_cur-d, num_nodes, residual_channels) 

            x = x.transpose(-3, -1)

            skip_cur = self.skip[k](x)
            if skip_out == None:
                skip_out = skip_cur
            else:
                skip_out = skip_out[..., -skip_cur.shape[-1]:] + skip_cur

            x = self.bns[k](x)

        # (batch_size, skip_channels, num_nodes, 1) 

        x = self.end1(self.end_tmp(skip_out))
        x = F.relu(x)
        x = x.transpose(-3, -1)

        # (batch_size, 1, num_nodes, out_timestamps) 

        x = self.end2(x)
        x = F.relu(x)

        # (batch_size, out_channels, num_nodes, out_timestamps) 

        x = x.transpose(-3, -1)
        return x
