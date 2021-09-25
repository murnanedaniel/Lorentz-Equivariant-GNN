from torch_geometric.nn import global_mean_pool
from .dynamic_edge_conv import DynamicEdgeConv

from torch import nn
import torch.nn.functional as F
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

        # The edge networks computes new edge features from connected nodes
        self.edge_networks = [
                make_mlp(
                (sum(self.input_dims[:i+1])) * 2,
                [self.hidden_dims[i]] * self.nb_layers,
                hidden_activation=hparams["activation"],
                output_activation=hparams["activation"],
                batch_norm=hparams["batch_norm"],
            ) for i in range(self.n_graph_iters)
        ]
        self.edge_convs = nn.ModuleList([
            DynamicEdgeConv(edge_network, k=hparams["k"], aggr="mean")
            for edge_network in self.edge_networks
        ])

        # The graph classifier outputs a final score (without sigmoid!)
#         output_size = sum(self.hidden_dims)
        output_size = sum(self.hidden_dims) + hparams["input_coords_dim"]
        
        self.graph_classifier = nn.Sequential(
            make_mlp(output_size, [256], output_activation=hparams["activation"]),
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
        convoluted_nodes = F.relu(torch.cat([input_features, convoluted_nodes], dim=-1))
#         all_features.append(convoluted_nodes)
        
        for i in range(1, self.n_graph_iters):
            convoluted_nodes_initial = convoluted_nodes
            convoluted_nodes = self.edge_convs[i](convoluted_nodes, x=convoluted_nodes, batch=batch.batch)
            convoluted_nodes = F.relu(torch.cat([convoluted_nodes, convoluted_nodes_initial], dim=-1))
            
#             all_features.append(convoluted_nodes)

#         all_features = torch.cat(all_features, dim=-1)
        all_features = convoluted_nodes

        global_average = global_mean_pool(all_features, batch.batch)       
        
        # Final layers
        output = self.graph_classifier(global_average)
        
        return output