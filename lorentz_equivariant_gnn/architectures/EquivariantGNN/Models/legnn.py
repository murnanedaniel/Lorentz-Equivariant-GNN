from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU

from torch import nn
import torch

from ..egnn_base import EGNNBase
from .model_utils import compute_radials, compute_initial_feature


class L_GCL(MessagePassing):
    """
    A PyGeometric MessagePassing class that computes a message along each edge, then updates 
    the x (co-ordinate) and h (features) node vectors based on these aggregated messages
    """
    
    def __init__(self, input_feature_dim, message_dim, output_feature_dim, edge_feature_dim, activation = nn.SiLU()):
        super().__init__(aggr='add') #  "Max" aggregation.
        
        radial_dim = 1  # Only one number is needed to specify Minkowski distance
        coordinate_dim = 4
        self.message_dim = message_dim

        # The MLP used to calculate messages
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * input_feature_dim + radial_dim + edge_feature_dim, message_dim),
            activation,
            nn.Linear(message_dim, message_dim),
            nn.Softsign()
            #activation
        )

        # The MLP used to update the feature vectors h_i
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_feature_dim + message_dim, message_dim),
            activation,
            nn.Linear(message_dim, output_feature_dim),
            nn.Softsign()
        )

        # Setup randomized weights
        self.layer = nn.Linear(message_dim, 1, bias = False)
        torch.nn.init.xavier_uniform_(self.layer.weight, gain = 0.001)

        # The MLP used to update coordinates (node embeddings) x_i
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            activation,
            self.layer
        )

        self.coordinate_linear_combination_mlp = nn.Linear(2 * coordinate_dim, coordinate_dim, bias = False)

    def forward(self, x, h, edge_index, edge_attribute = None):
        
        radial, _ = compute_radials(edge_index, x)

        return self.propagate(edge_index, x=x, h=h, radial=radial)

    def message(self, x_i, x_j, h_i, h_j, radial):

        
        h_messages = self.compute_messages(h_i, h_j, radial)
        x_messages = (x_i - x_j)*self.coordinate_mlp(h_messages)
        
        return torch.cat([h_messages, x_messages], axis=1)
    
    
    def update(self, aggr, x, h):
        h_next = self.feature_mlp(torch.cat([aggr[:, :self.message_dim], h], axis=1) )
        x_next = x + aggr[:, self.message_dim:]
    
        return h_next, x_next
    
    
    def compute_messages(self, source, target, radial, edge_attribute = None):
        """
        Calculates the messages to send between two nodes 'target' and 'source' to be passed through the network.
        The message is computed via an MLP of Lorentz invariants.

        :param source: The source node's feature vector h_i
        :param target: The target node's feature vector h_j
        :param radial: The Minkowski distance between the source and target's coordinates
        :param edge_attribute: Features at the edge connecting the source and target nodes
        :return: The message m_{ij}
        """
        
        if edge_attribute is None:
            message_inputs = torch.cat([source, target, radial], dim = 1)  # Setup input for computing messages through MLP
        else:
            message_inputs = torch.cat([source, target, radial, edge_attribute], dim = 1)  # Setup input for computing messages through MLP

        out = self.edge_mlp(message_inputs)  # Apply \phi_e to calculate the messages
        return out

    

class LEGNN(EGNNBase):
    """
    The main network used for Lorentz group equivariance consisting of several layers of L_GCLs
    """

    def __init__(self, hparams):
        """
        Sets up the equivariant network and creates the necessary L_GCL layers

        :param input_feature_dim: The amount of numbers needed to specify a feature inputted into the LEGNN
        :param message_dim: The amount of numbers needed to specify a message passed through the LEGNN
        :param output_feature_dim: The amount of numbers needed to specify the updated feature after passing through the LEGNN
        :param edge_feature_dim: The amount of numbers needed to specify an edge attribute a_{ij}
        :param device: Specification on whether the cpu or gpu is to be used
        :param activation: The activation function used as the main non-linearity throughout the LEGNN
        :param n_layers: The number of layers the LEGNN network has
        """

        super().__init__(hparams)
        self.message_dim = hparams["message_dim"]
        self.activation = getattr(nn, hparams["activation"])
        self.n_layers = hparams["n_layers"]
        self.feature_in = nn.Linear(hparams["input_feature_dim"], self.message_dim)  # Initial mixing of features
        self.feature_out = nn.Linear(self.message_dim, hparams["output_feature_dim"])  # Final mixing of features to yield desired output

        for i in range(hparams["n_layers"]):
            self.add_module("gcl_%d" % i, )
        self.equivariant_layers = nn.ModuleList([
            L_GCL(self.message_dim, 
                  self.message_dim, 
                  self.message_dim,
                  hparams["edge_feature_dim"], 
                  activation = self.activation())
            for _ in range()
        ])

    def forward(self, x, edges, edge_attribute = None):
        
        h = compute_initial_feature(edges, x)
        h = self.feature_in(h.unsqueeze(1))
        
        for _ in range(self.n_layers):
            h, x = self._modules["gcl_%d" % i](x, h, edges, edge_attribute = edge_attribute)
        
        h = self.feature_out(h)
        return h, x
    