from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import global_mean_pool

from torch import nn
import torch
import torch.nn.functional as F
from torch_cluster import knn_graph

from ..egnn_base import EGNNBase
from ..utils import make_mlp, get_fully_connected_edges
from .model_utils import compute_vector_invariants, get_minkowski_distance, get_scalar_product
from .general_gnn import GeneralGNN
   

class SkipGeneralGNN(GeneralGNN):
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
        
        self.convolution_layers = nn.ModuleList(
            [GeneralConv(
                  scalar_dim = len(self.scalars) + hparams["scalar_dim"], 
                  message_dim = self.message_dim[0], 
                  invariant_vector_dim = hparams["invariant_vector_dim"], 
                  vector_dim = hparams["vector_dim"],
                  hparams = hparams)] 
            + [GeneralConv(
                  scalar_dim = sum(self.message_dim[:i]) + len(self.scalars) + hparams["scalar_dim"], 
                  message_dim = self.message_dim[i], 
                  invariant_vector_dim = hparams["invariant_vector_dim"], 
                  vector_dim = self.propagate_vector_dims,
                  hparams = hparams)
                for i in range(1, self.n_graph_iters)
            ])
        
        fully_connected_dim = sum(self.message_dim) + len(self.scalars) + hparams["scalar_dim"]
        
        # The graph classifier outputs a final score (without sigmoid!)
        self.graph_classifier = nn.Sequential(
            make_mlp(fully_connected_dim, [hparams["final_dim"]]),
            nn.Dropout(hparams["dropout"]),
            make_mlp(hparams["final_dim"], [1], output_activation=None))
                
    def forward(self, batch):
        
        s, v = self.get_node_features(batch)
        
        for i in range(self.n_graph_iters):                
            
            s_initial = s # For skip connection
            s, v = self.convolution_layers[i](s, v, batch.edge_index, edge_attribute = None)
            s = F.relu(torch.cat([s, s_initial], dim=-1))
            
            if self.graph_construction == "dynamic_knn":
                batch.edge_index = knn_graph(s, self.k, batch.batch)
        
        global_average = global_mean_pool(s, batch.batch)
        
        # Final layers
        output = self.graph_classifier(global_average)
        
        return output
    