import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_layers=3,
                 dropout=0.5, relu_first = True):
        super(MLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.relu_first = relu_first

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            if self.relu_first:
                x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            if not self.relu_first:
                x = F.relu(x, inplace=True)


            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return F.log_softmax(x, dim=-1)



class MLPLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPLinear, self).__init__()
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        return F.log_softmax(self.lin(x), dim=-1)
