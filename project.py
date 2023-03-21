from functools import reduce
from operator import mul
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        
        # print("Before cat", len(out), out[0].shape)

        h = torch.cat(out,dim=1)
        # print("h:", h.shape)
        h = self.mlp(h)
        # print("after mlp", h.shape)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

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
        # self.gcn_conv = gcn(dilation_channels,in_channels,dropout,support_len=3)
        
        if out_channels != None:
            self.out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        else:
            self.out = None
        
    def forward(self, input, edge_index, supports=None, edge_weight=None):
        # (batch_size, t, num_nodes, in_channels)
        input = input.transpose(-3, -1)

        g1 = self.gate1(input)
        g2 = self.gate2(input)

        # (batch_size, dilation_channels, num_nodes, t_cur - d[i])

        g1 = F.tanh(g1)
        g2 = F.sigmoid(g2)
        out1 = g1 * g2

        # gcn_out = self.gcn_conv(g1 * g2, supports)

        # (batch_size, t_cur - d[i], num_nodes, dilation_channels)
        g1 = g1.transpose(-3, -1)
        g2 = g2.transpose(-3, -1)
        
        out2 = torch.flatten(g1 * g2, end_dim=-3)
        edge_index = edge_index.expand(out2.size(0), edge_index.size(-2), edge_index.size(-1))
        edge_weight = edge_weight.expand(out2.size(0), edge_weight.size(-1)).reshape((out2.size(0), edge_weight.size(-1), 1))

        dataset = [Data(x=x_input, edge_index=e_i, edge_attr=e_w) for x_input, e_i, e_w in zip(out2, edge_index, edge_weight)]

        loader = DataLoader(dataset=dataset, batch_size=out2.size(0))

        data = next(iter(loader))


        gcn_out = None
        cur_shape = list(g1.shape)
        batch_size = reduce(mul, cur_shape[:-2])

        g1 = torch.reshape(g1, (batch_size, cur_shape[-2], cur_shape[-1]))
        g2 = torch.reshape(g2, (batch_size, cur_shape[-2], cur_shape[-1]))



        """for batch in range(len(g1)): 
            g = self.gcn_conv(g1[batch] * g2[batch], edge_index, edge_weight)
            if gcn_out == None:
                gcn_out = torch.unsqueeze(g, 0)
            else:
                gcn_out = torch.cat([gcn_out, torch.unsqueeze(g, 0)])"""

        gcn_out = self.gcn_conv(data.x, data.edge_index, edge_weight=torch.flatten(data.edge_attr))

        gcn_out = torch.reshape(gcn_out, tuple(cur_shape[:-1] + [self.gcn_conv.out_channels]))
        gcn_out = F.dropout(gcn_out, p=self.dropout)

        
        if self.out != None:
            gcn_out = gcn_out.transpose(-3, -1)
            gcn_out = self.out(gcn_out)
            gcn_out = gcn_out.transpose(-3, -1)  

        


        # (batch_size, t_cur - d[i], num_nodes, in_channels)

        

        # (batch_size, out_channels, num_nodes, t-dilation)

        return out1, gcn_out
    
class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, adj, in_channels, out_channels, out_timestamps, dilations=[1,2,1,2,1,2,1,2], dropout=0.3, residual_channels=32, dilation_channels=32, skip_channels=256, supports=None):
        super(GraphWaveNet, self).__init__()

        self.total_dilation = sum(dilations)
        self.input = nn.Conv2d(in_channels=in_channels, out_channels=residual_channels, kernel_size=(1, 1))

        self.adj = adj
        self.num_nodes = num_nodes
        self.supports = supports

        self.e1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.e2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)

        self.spatial_temporals = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bns2 = nn.BatchNorm2d(skip_channels)
        self.skip = nn.ModuleList()
        
        for d in dilations:
            self.spatial_temporals.append(SpatialTemporal(residual_channels, d, dilation_channels=dilation_channels, dropout=dropout))
            self.skip.append(nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1)))
            self.bns.append(nn.BatchNorm2d(residual_channels))
            

        self.end_tmp1 = nn.Conv2d(in_channels=skip_channels, out_channels=512, kernel_size=(1, 1))
        self.end_tmp2 = nn.Conv2d(in_channels=512, out_channels=out_timestamps, kernel_size=(1, 1))
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
        if self.total_dilation > x.shape[-1]:
            x = F.pad(x, (self.total_dilation - x.shape[-1] + 1, 0))
        x = self.input(x)

        skip_out = None

        # (batch_size, residual_channels, num_nodes, t)
        for k in range(len(self.spatial_temporals)):
            residual = x

            x = x.transpose(-3, -1)
            # (batch_size, t_cur, num_nodes, residual_channels)
            out1, x = self.spatial_temporals[k](x, edge_list, supports=self.supports + [adp], edge_weight=edge_weight)

            # (batch_size, t_cur-d, num_nodes, residual_channels) 

            x = x.transpose(-3, -1)
            x = x + residual[..., -x.shape[-1]:]
            x = self.bns[k](x)

            skip_cur = self.skip[k](out1)
            if skip_out == None:
                skip_out = skip_cur
            else:
                skip_out = skip_out[..., -skip_cur.shape[-1]:] + skip_cur
            


        # (batch_size, skip_channels, num_nodes, 1) 

        x = (self.end_tmp2(F.relu(self.end_tmp1(F.relu(skip_out)))))
        """x = F.relu(x)
        x = x.transpose(-3, -1)

        # (batch_size, 1, num_nodes, out_timestamps) 

        x = self.end2(x)
        x = F.relu(x)

        # (batch_size, out_channels, num_nodes, out_timestamps)
        x = x.transpose(-3, -1)"""

        
        return x






    