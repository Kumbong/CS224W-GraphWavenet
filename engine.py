import torch.optim as optim
from model import *
from project import *
import util
from torchviz import make_dot
import os
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit, adj_mx):
        # self.model = gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model = GraphWaveNet(num_nodes, 2, 1, 12)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

        self.edge_index = [[], []]
        self.edge_weight = []
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx.item((i, j)) != 0:
                    self.edge_index[0].append(i)
                    self.edge_index[1].append(j)
                    self.edge_weight.append(adj_mx.item((i, j)))

        self.edge_index = torch.tensor(self.edge_index)
        self.edge_weight = torch.tensor(self.edge_weight)

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = input.transpose(-3, -1)
        output = self.model(input, self.edge_index, self.edge_weight)

        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        input = input.transpose(-3, -1)
        output = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
