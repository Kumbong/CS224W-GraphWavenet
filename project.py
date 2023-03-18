import torch
from torch import nn
from torch_geometric.nn import GCNConv

class SpatialTemporal(nn.Module):
    # in_channels=32
    def __init__(self, in_channels, dilations, out_channels=None, dilation_channels=32, dropout=0.3):
        self.gate1 = nn.ModuleList()
        self.gate2 = nn.ModuleList()

        for d in dilations:
            

        nn.Conv2d(in_channels=in_channels, out_channels=dilation_channels, kernel_size=(1,1))
        self.gate2 = nn.Conv2d(in_channels=in_channels, out_channels=dilation_channels, kernel_size=(1,1))
    
        if out_channels == None:
            self.gcn_conv = GCNConv(in_channels=dilation_channels, out_channels=in_channels)
        else:
            self.gcn_conv = GCNConv(in_channels=dilation_channels, out_channels=out_channels)
        
    def forward(self, input, edge_index, edge_weight=None):

    