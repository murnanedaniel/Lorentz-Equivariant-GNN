from torch import nn
import torch


class L_GCL(nn.Module):
    """
    SO+(1, 3) Equivariant Convolution Layer
    """

    def __init__(self, input_feature_dim, message_dim, output_feature_dim, edge_feature_dim, activation = nn.SiLU()):
        super(L_GCL, self).__init__()
        radial_dim = 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * input_feature_dim + radial_dim + edge_feature_dim, message_dim),
            activation,
            nn.Linear(message_dim, message_dim),
            activation
        )

        self.feature_mlp = nn.Sequential(
            nn.Linear(input_feature_dim + message_dim, message_dim),
            activation,
            nn.Linear(message_dim, output_feature_dim)
        )

        layer = nn.Linear(message_dim, 1, bias = False)
        torch.nn.init.xavier_uniform_(layer.weight, gain = 0.001)

        self.coordinate_mlp = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            activation,
            layer
        )

    def compute_messages(self, source, target, radial, edge_attribute):
        #message_inputs = torch.cat([source, target, radial], dim = 1)
        message_inputs = torch.cat([source, target, radial, edge_attribute], dim = 1)
        out = self.edge_mlp(message_inputs)
        return out

    def update_feature_vectors(self, h, edge_index, messages):
        row, col = edge_index
        message_aggregate = unsorted_segment_sum(messages, row, num_segments = h.size(0))
        feature_inputs = torch.cat([h, message_aggregate], dim = 1)
        out = self.feature_mlp(feature_inputs)
        return out, message_aggregate

    def update_coordinates(self, x, edge_index, coordinate_difference, messages):
        row, col = edge_index
        weighted_differences = coordinate_difference * self.coordinate_mlp(messages)
        relative_updated_coordinates = unsorted_segment_sum(weighted_differences, row, num_segments = x.size(0))
        x += relative_updated_coordinates
        return x

    def compute_radials(self, edge_index, x):
        row, col = edge_index
        coordinate_differences = x[row] - x[col]
        radial = torch.sum(coordinate_differences ** 2, 1).unsqueeze(1)
        return radial, coordinate_differences

    def forward(self, h, x, edge_index, edge_attribute = None):
        row, col = edge_index
        radial, coordinate_differences = self.compute_radials(edge_index, x)

        messages = self.compute_messages(h[row], h[col], radial, edge_attribute)
        x_updated = self.update_coordinates(x, edge_index, coordinate_differences, messages)
        h_updated, _ = self.update_feature_vectors(h, edge_index, messages)

        return h_updated, x_updated


class LEGNN(nn.Module):
    def __init__(self, input_feature_dim, message_dim, output_feature_dim, edge_feature_dim,
                 device = 'cpu', activation = nn.SiLU(), n_layers = 4):
        super(LEGNN, self).__init__()
        self.message_dim = message_dim
        self.device = device
        self.n_layers = n_layers
        self.feature_in = nn.Linear(input_feature_dim, message_dim)
        self.feature_out = nn.Linear(message_dim, output_feature_dim)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, L_GCL(self.message_dim, self.message_dim, self.message_dim,
                                                edge_feature_dim, activation = activation))
        self.to(self.device)

    def forward(self, h, x, edges, edge_attribute):
        h = self.feature_in(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["gcl_%d" % i](h, x, edges, edge_attribute = edge_attribute)
        h = self.feature_out(h)
        return h, x


"""
This method is used to compute the message aggregation for 'sum'
"""
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


"""
Returns an array of edge links corresponding to a fully-connected graph
"""
def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


"""
Creates an extended version of get_edges(n_nodes) whereby an arbitrary number of batches may be created.
The extension occurs by allowing for several graphs to be input into the system, and each edge is labeled
by a random permutation and offset by a factor depending on which graph is currently selected.
"""
def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1) # Create 1D-tensor of 1s for each edge for a batch of graphs
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])] # Convert 2D array of edge links to a 2D-tensor
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i) # Offset rows for each graph in the batch
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr


if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3

    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size *  n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    # Initialize LEGNN
    legnn = LEGNN(input_feature_dim = n_feat, message_dim = 32, output_feature_dim = 1, edge_feature_dim = 1)

    # Run LEGNN
    h, x = legnn(h, x, edges, edge_attr)