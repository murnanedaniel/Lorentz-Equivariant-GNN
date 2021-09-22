from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import global_mean_pool

from torch import nn
import torch

from ..egnn_base import EGNNBase
from ..utils import make_mlp
from .model_utils import compute_radials, compute_initial_feature


class L_GCL(MessagePassing):
    """
    A PyGeometric MessagePassing class that computes a message along each edge, then updates 
    the x (co-ordinate) and h (features) node vectors based on these aggregated messages
    """
    
    def __init__(self, input_feature_dim, message_dim, output_feature_dim, edge_feature_dim, coordinate_dim, nb_edge_layers, nb_node_layers, activation):
        super().__init__(aggr='mean') #  "Max" aggregation.
        
        self.message_dim = message_dim
        self.input_feature_dim = input_feature_dim

        # The MLP used to calculate messages        
        self.edge_mlp = make_mlp(2 * input_feature_dim + edge_feature_dim,
                                 [message_dim]*nb_edge_layers + [1], 
                                 hidden_activation=activation,
                                output_activation="Softsign")

        # The MLP used to update the feature vectors h_i
        self.feature_mlp = make_mlp(input_feature_dim,
                                 [message_dim]*nb_node_layers, hidden_activation=activation)
        

    def forward(self, x, h, edge_index, edge_attribute = None):
       
        return self.propagate(edge_index, x=x, h=h)

    def message(self, x_i, x_j, h_i, h_j):

        h_messages = self.compute_messages(h_i, h_j, x_i, x_j)
        h_score = self.edge_mlp(h_messages)
        
        weighted_h = h_score * h_j
        x_messages = (x_i - x_j)*h_score
        
        return torch.cat([weighted_h, x_messages], axis=1)
    
    def update(self, aggr, x, h):
                        
        if h.shape[1] == self.message_dim:
            h_next = h + self.feature_mlp(aggr[:, :self.input_feature_dim])
        else:
            h_next = self.feature_mlp(aggr[:, :self.input_feature_dim])
        x_next = x + aggr[:, self.input_feature_dim:]
    
        return h_next, x_next
    
    def compute_messages(self, h_i, h_j, x_i, x_j):
        """
        Calculates the messages to send between two nodes 'target' and 'source' to be passed through the network.
        The message is computed via an MLP of Lorentz invariants.

        :param source: The source node's feature vector h_i
        :param target: The target node's feature vector h_j
        :param radial: The Minkowski distance between the source and target's coordinates
        :param edge_attribute: Features at the edge connecting the source and target nodes
        :return: The message m_{ij}
        """
        distance = self.get_minkowski_distance(x_i, x_j)
        scalar_product = self.get_scalar_product(x_i, x_j)
        
        messages = torch.cat([h_i, h_j, distance, scalar_product], dim = 1)  # Setup input for computing messages through MLP

        return messages
    
    def get_minkowski_distance(self, x_i, x_j):
        """
        Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j

        :param edge_index: Array containing the connection between nodes
        :param x: The coordinates (node embeddings)
        :return: Minkowski distances (squared) and coordinate differences x_i - x_j
        """

        coordinate_differences = x_i - x_j
        coordinate_differences_squared = coordinate_differences ** 2
        coordinate_differences_squared[:, 0] = -coordinate_differences_squared[:, 0]  # Place minus sign on time coordinate as \eta = diag(-1, 1, 1, 1)
        minkowski_distance_squared = torch.sum(coordinate_differences_squared, 1).unsqueeze(1)
        
        return minkowski_distance_squared
    
    def get_scalar_product(self, x_i, x_j):
        
        product = x_i * x_j
        product[:, 0] = -product[:, 0]
        scalar_product = torch.sum(product, 1).unsqueeze(1)
        
        return scalar_product

    

class AttentionLEGNN(EGNNBase):
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
        torch.autograd.set_detect_anomaly(True)
        self.message_dim = hparams["message_dim"]
        self.activation = hparams["activation"]
        self.n_graph_iters = hparams["n_graph_iters"]

        self.initial_equivariant_layer = L_GCL(hparams["input_feature_dim"], 
                                              self.message_dim, 
                                              self.message_dim,
                                              hparams["edge_feature_dim"], 
                                              hparams["coordinate_dim"],
                                              hparams["nb_edge_layers"],
                                              hparams["nb_node_layers"],
                                              activation = self.activation)
        
        self.equivariant_layers = nn.ModuleList([
            L_GCL(self.message_dim, 
                  self.message_dim, 
                  self.message_dim,
                  hparams["edge_feature_dim"], 
                  hparams["coordinate_dim"],
                  hparams["nb_edge_layers"],
                  hparams["nb_node_layers"],
                  activation = self.activation)
            for _ in range(self.n_graph_iters - 1)
        ])
        
        # The graph classifier outputs a final score (without sigmoid!)
        self.graph_classifier = nn.Sequential(
            make_mlp(self.message_dim, [self.message_dim]),
            nn.Dropout(hparams["dropout"]),
            make_mlp(self.message_dim, [1], output_activation=None))
    
    def forward(self, batch):
        
        x = batch.x.float()
        h = compute_initial_feature(x).float().unsqueeze(1)
        
        h, x = self.initial_equivariant_layer(x, h, batch.edge_index, edge_attribute = None)
        
        for i in range(self.n_graph_iters - 1):
            h, x = self.equivariant_layers[i](x, h, batch.edge_index, edge_attribute = None)
        
        global_average = global_mean_pool(h, batch.batch)
        
        # Final layers
        output = self.graph_classifier(global_average)
        
        return output
    