from torch import nn
import torch

from ..egnn_base import EGNNBase


class L_GCL(nn.Module):
    """
    SO+(1, 3) Equivariant Convolution Layer
    """

    def __init__(self, input_feature_dim, message_dim, output_feature_dim, edge_feature_dim, activation = nn.SiLU()):
        """
        Sets up the MLPs needed to compute the layer update of the equivariant network.

        :param input_feature_dim: The amount of numbers needed to specify a feature inputted into the GCL
        :param message_dim: The amount of numbers needed to specify a message passed through the GCL
        :param output_feature_dim: The amount of numbers needed to specify the updated feature after passing through the GCL
        :param edge_feature_dim: The amount of numbers needed to specify an edge attribute a_{ij}
        :param activation: The activation function used as the main non-linearity throughout the GCL
        """

        super(L_GCL, self).__init__()
        radial_dim = 1  # Only one number is needed to specify Minkowski distance
        coordinate_dim = 4

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

    def update_feature_vectors(self, h, edge_index, messages):
        """
        Updates the feature vectors via an MLP of Lorentz invariants, specifically the feature vector itself and
        aggregated messages.

        :param h: The feature vectors outputted from the previous layer
        :param edge_index: Array containing the connection between nodes
        :param messages: List of messages m_{ij} used to calculated an aggregated message for h
        :return: The updated feature vectors h_i^{l+1}
        """

        row, col = edge_index
        message_aggregate = unsorted_segment_sum(messages, row, num_segments = h.size(0))
        feature_inputs = torch.cat([h, message_aggregate], dim = 1)
        out = self.feature_mlp(feature_inputs)
        return out, message_aggregate

    def update_coordinates(self, x, edge_index, coordinate_difference, messages):
        """
        Updates the coordinates (node embeddings) through the update rule
            x_i^{l+1} = x_i^l + Σ(x_i^l - x_j^l)\phi_x(m_{ij})

        :param x: The coordinates (node embeddings) outputted from the previous layer
        :param edge_index: Array containing the connection between nodes
        :param coordinate_difference: The differences between two coordinates x_i and x_j
        :param messages: List of messages m_{ij} to be passed through the coordinate MLP \phi_x
        :return: The updated coordinates (node embeddings) x_i^{l+1}
        """

        row, col = edge_index

        linear_input = torch.cat([x[row], x[col]], dim = 1)
        coordinate_linear_combination = self.coordinate_linear_combination_mlp(linear_input)

        #print(messages)
        #print(self.coordinate_mlp(messages))
        #weighted_differences = coordinate_difference * self.coordinate_mlp(messages)  # Latter part of the update rule
        weighted_linear_combination = coordinate_linear_combination * self.coordinate_mlp(messages)  # Latter part of the update rule
        relative_updated_coordinates = unsorted_segment_mean(weighted_linear_combination, row, num_segments = x.size(0))  # Computes the summation
        x += relative_updated_coordinates  # Finishes the update rule
        return x

    @staticmethod
    def compute_radials(edge_index, x):
        """
        Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j

        :param edge_index: Array containing the connection between nodes
        :param x: The coordinates (node embeddings)
        :return: Minkowski distances (squared) and coordinate differences x_i - x_j
        """

        row, col = edge_index
        coordinate_differences = x[row] - x[col]
        minkowski_distance_squared = coordinate_differences ** 2
        minkowski_distance_squared[:, 0] = -minkowski_distance_squared[:, 0]  # Place minus sign on time coordinate as \eta = diag(-1, 1, 1, 1)
        radial = torch.sum(minkowski_distance_squared, 1).unsqueeze(1)
        return radial, coordinate_differences

    def forward(self, h, x, edge_index, edge_attribute = None):
        row, col = edge_index
        radial, coordinate_differences = self.compute_radials(edge_index, x)

        messages = self.compute_messages(h[row], h[col], radial, edge_attribute)
        x_updated = self.update_coordinates(x, edge_index, coordinate_differences, messages)
        h_updated, _ = self.update_feature_vectors(h, edge_index, messages)

        return h_updated, x_updated


class LEGNN(EGNNBase):
    """
    The main network used for Lorentz group equivariance consisting of several layers of L_GCLs
    """

    def __init__(self, hparams):
#     input_feature_dim, message_dim, output_feature_dim, edge_feature_dim,
#                  device = 'cpu', activation = nn.SiLU(), n_layers = 4):
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

        for i in range(0, hparams["n_layers"]):
            self.add_module("gcl_%d" % i, L_GCL(self.message_dim, self.message_dim, self.message_dim,
                                                hparams["edge_feature_dim"], activation = self.activation()))

    def forward(self, x, edges, edge_attribute = None):
        
        h, _ = self.compute_radials(edges, x)  # torch.zeros(n_nodes, 1)
        
        h = self.feature_in(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](h, x, edges, edge_attribute = edge_attribute)
        h = self.feature_out(h)
        return h, x
    
    @staticmethod
    def compute_radials(edge_index, x):
        """
        Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j

        :param edge_index: Array containing the connection between nodes
        :param x: The coordinates (node embeddings)
        :return: Minkowski distances (squared) and coordinate differences x_i - x_j
        """

        row, col = edge_index
        coordinate_differences = x[row] - x[col]
        minkowski_distance_squared = coordinate_differences ** 2
        minkowski_distance_squared[:, 0] = -minkowski_distance_squared[:, 0]  # Place minus sign on time coordinate as \eta = diag(-1, 1, 1, 1)
        radial = torch.sum(minkowski_distance_squared, 1).unsqueeze(1)
        return radial, coordinate_differences


"""
This method is used to compute the message aggregation for 'sum'
"""
def unsorted_segment_sum(data, segment_ids, num_segments):
    device = data.device
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result = result.to(device)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    segment_ids = segment_ids.to(device)
    result.scatter_add_(0, segment_ids, data)
    return result


"""
This method is used to compute the message aggregation for 'mean'
"""
def unsorted_segment_mean(data, segment_ids, num_segments):
    device = data.device
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    segment_ids = segment_ids.to(device)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result = result.to(device)
    count = data.new_full(result_shape, 0)
    count = count.to(device)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


"""
Returns an array of edge links corresponding to a fully-connected graph
"""
def get_edges(n_nodes):
    
    node_list = torch.arange(n_nodes)
    edges = torch.combinations(node_list, r=2).T
    bidirectional_edges = torch.cat([edges, edges.flip(0)], axis=1)
    
    return bidirectional_edges



"""
Creates an extended version of get_edges(n_nodes) whereby an arbitrary number of batches may be created.
The extension occurs by allowing for several graphs to be input into the system, and each edge is labeled
by a random permutation and offset by a factor depending on which graph is currently selected.
"""
def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)  # Create 1D-tensor of 1s for each edge for a batch of graphs
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]  # Convert 2D array of edge links to a 2D-tensor
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)  # Offset rows for each graph in the batch
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 1#8
    n_nodes = 4
    n_feat = 1
    x_dim = 4

    # Dummy variables h, x and fully connected edges
    h = torch.rand(batch_size * n_nodes, n_feat)
    x = torch.rand(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    print(x)

    # Initialize LEGNN
    legnn = LEGNN(input_feature_dim = n_feat, message_dim = 32, output_feature_dim = 1, edge_feature_dim = 1)

    # Run LEGNN
    h, x = legnn(h, x, edges, edge_attr)
    print(x)