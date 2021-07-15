import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.typing import Adj, Size, OptTensor, Tensor


class LEGNN_Network(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(LEGNN_Network, self).__init__(aggr = 'add')
        self.message_mlp = Sequential(Linear(in_channels, out_channels),
                                      ReLU(),
                                      Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x = x)

    def message(self, x_i, x_j):
        distance_squared = (x_i - x_j) ** 2
        distance_squared = torch.sum(distance_squared, dim = 1).unsqueeze(1)
        return self.message_mlp(distance_squared)


if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                               [1, 0]], dtype = torch.long)
    x = torch.tensor([[-1, 1, 1, 1],
                      [0, 2, 3, 4]], dtype = torch.float)
    data = Data(x = x, edge_index = edge_index)

    model = LEGNN_Network(1, 1)
    print(model(x, edge_index))
