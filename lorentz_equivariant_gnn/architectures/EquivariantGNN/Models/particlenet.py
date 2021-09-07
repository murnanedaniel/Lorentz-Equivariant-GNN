from torch import nn
import torch

from ..egnn_base import EGNNBase

class EdgeConv(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        nb_layers,
        hidden_activation="ReLU",
        batch_norm=True,
    ):
        super(EdgeConv, self).__init__()
        self.network = make_mlp(
            input_dim * 2,
            [hidden_dim] * nb_layers,
            hidden_activation=hidden_activation,
            output_activation="ReLU",
            batch_norm=batch_norm,
        )

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        
        messages = scatter_mean(x[start] - x[end], end, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([x, messages], dim=1)
        
        return self.network(node_inputs).squeeze()
    
class ParticleNet(EGNNBase):
    """
    A vanilla GCN that simply convolves over a fully-connected graph
    """


    def __init__(self, hparams):
        super(, self).__init__(hparams)
        
        self.n_graph_iters = hparams["n_graph_iters"]
        self.hidden_dim = hparams["hidden_dim"]
        self.nb_layers = hparams["nb_layers"]

        # The node encoder transforms input node features to the hidden space
        self.node_encoder = make_mlp(hparams["input_dim"], [self.hidden_dim]*self.nb_layers)

        # The edge network computes new edge features from connected nodes
        self.edge_conv = EdgeConv(self.hidden_dim, self.hidden_dim, self.nb_layers, batch_norm=hparams["batch_norm"])

        # The edge classifier computes final edge scores
        self.graph_classifier = make_mlp(self.hidden_dim,
                                        [self.hidden_dim, 2],
                                        output_activation=None)
        
        # Graph building options
        self.k_max = k_max

        
    def forward(self, x, edges, edge_attribute = None):

        # 1. Encode features
        convoluted_nodes = self.node_encoder(x).squeeze()
        
        # 2. Run edge conv x 3
        for i in range(self.n_graph_iters):
            skip_connection = convoluted_nodes
            convoluted_nodes = self.edge_conv(convoluted_nodes, edges)
            convoluted_nodes += skip_connection
        
        # 4. Apply global average
        global_average = convoluted_nodes.mean(dim=0)
        
        # 5. Apply FC classifier
        classified_output = self.graph_classifier(global_average)
        
        return classified_output