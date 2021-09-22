from torch_geometric.nn import global_mean_pool
from .dynamic_edge_conv import DynamicEdgeConv

from torch import nn
import torch

from ..utils import make_mlp
from ..egnn_base import EGNNBase

    
class ParticleNet(EGNNBase):
    """
    A vanilla GCN that simply convolves over a fully-connected graph
    """


    def __init__(self, hparams):
        super().__init__(hparams)
        
        self.n_graph_iters = len(hparams["message_dim"])
        self.hidden_dims = hparams["message_dim"]
        self.input_dims = [hparams["input_coords_dim"]] + self.hidden_dims[:-1]
        self.nb_layers = hparams["nb_layers"]
        
        # The node encoder transforms input node features to the hidden space
        self.node_encoder = make_mlp(hparams["input_coords_dim"], [self.hidden_dims[0]]*self.nb_layers)

        # The edge networks computes new edge features from connected nodes
        self.edge_networks = [
                make_mlp(
                input_dim * 2,
                [hidden_dim] * self.nb_layers,
                hidden_activation=hparams["activation"],
                output_activation=hparams["activation"],
                batch_norm=hparams["batch_norm"],
            ) for input_dim, hidden_dim in zip(self.input_dims, self.hidden_dims)
        ]
        self.edge_convs = nn.ModuleList([
            DynamicEdgeConv(edge_network, k=hparams["k"], aggr="mean")
            for edge_network in self.edge_networks
        ])

        # The graph classifier outputs a final score (without sigmoid!)
        self.graph_classifier = nn.Sequential(
            make_mlp(sum(self.hidden_dims), [256]),
            nn.Dropout(hparams["dropout"]),
            make_mlp(256, [1], output_activation=None)
        )


    def concat_feature_set(self, batch, feature_set):
        
        return torch.cat([batch[feature].unsqueeze(1) 
                   if len(batch[feature].shape)==1 
                   else batch[feature] for feature in self.hparams[feature_set]], dim=-1)
        
    def forward(self, batch):
        
        all_features = []
        
        input_features = self.concat_feature_set(batch, "features").float()
        
        input_coordinates = self.concat_feature_set(batch, "coordinates").float()
        
        convoluted_nodes = self.edge_convs[0](input_features, x=input_coordinates, batch=batch.batch)
        all_features.append(convoluted_nodes)
        
        for i in range(1, self.n_graph_iters):
            convoluted_nodes = self.edge_convs[i](convoluted_nodes, x=convoluted_nodes, batch=batch.batch)
            all_features.append(convoluted_nodes)

        all_features = torch.cat(all_features, dim=-1)

        global_average = global_mean_pool(all_features, batch.batch)       
        
        # Final layers
        output = self.graph_classifier(global_average)
        
        return output