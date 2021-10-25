from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import global_mean_pool

from torch import nn
import torch
from torch_cluster import knn_graph

from ..egnn_base import EGNNBase
from ..utils import make_mlp, get_fully_connected_edges
from .model_utils import compute_vector_invariants, get_minkowski_distance, get_scalar_product


class GeneralConv(MessagePassing):
    """
    A PyGeometric MessagePassing class that computes a message along each edge, then updates 
    the x (co-ordinate) and h (features) node vectors based on these aggregated messages
    """
    
    def __init__(self, 
                 scalar_dim, 
                 message_dim, 
                 invariant_vector_dim, 
                 vector_dim, 
                 hparams):
        super().__init__(aggr='mean') #  "Max" aggregation.
        
        self.message_dim = message_dim
        self.scalar_dim = scalar_dim
        self.equivariant = hparams["equivariant"]
        self.nb_node_layers = hparams["nb_node_layers"]

        # The MLP used to calculate messages
        edge_feature_size =  (2*scalar_dim+invariant_vector_dim) if self.equivariant else (2*scalar_dim+2*vector_dim)
        self.edge_mlp = make_mlp(edge_feature_size,
                                 [message_dim]*hparams["nb_edge_layers"], 
                                 hidden_activation=hparams["activation"],
                                output_activation=hparams["activation"])
        
        # The MLP used to calculate messages        
        if self.equivariant:
            self.attention_mlp = make_mlp(message_dim, [message_dim]*hparams["nb_edge_layers"]+ [1], 
                                     hidden_activation=hparams["activation"],
                                    output_activation="Softsign")

        # The MLP used to update the feature vectors h_i
        if self.nb_node_layers > 0:
            self.feature_mlp = make_mlp(scalar_dim + message_dim,
                                 [message_dim]*self.nb_node_layers, hidden_activation=hparams["activation"])
        

    def forward(self, s, v, edge_index, edge_attribute = None):
        
#         if self.equivariant:
#             return self.propagate(edge_index, s=s, v=v)
#         else:
#             return self.propagate(edge_index, s=s, v=None)
        return self.propagate(edge_index, s=s, v=v)

    def message(self, s_i, s_j, v_i, v_j):

        edge_inputs = self.get_edge_inputs(s_i, s_j, v_i, v_j)
        s_messages = self.edge_mlp(edge_inputs)
    
        if self.equivariant:
            v_messages = (v_i - v_j)*self.attention_mlp(s_messages)
            
            return torch.cat([s_messages, v_messages], axis=1)
        
        else:
            return s_messages
    
    def update(self, aggr, s, v):
        
        if self.nb_node_layers > 0:
            s_next = self.feature_mlp(torch.cat([aggr[:, :self.message_dim], s], axis=1) )
        else:
            s_next = aggr[:, :self.message_dim]
        
        if self.equivariant:
            v_next = v + aggr[:, self.message_dim:]
        else:
            v_next = None
        
        return s_next, v_next
    
    def get_edge_inputs(self, s_i, s_j, v_i, v_j):
        
        if self.equivariant:
            distance = get_minkowski_distance(v_i, v_j)
            scalar_product = get_scalar_product(v_i, v_j)
            #edge_inputs = torch.cat([s_i, s_i - s_j, distance, scalar_product], dim = 1)  # Setup input for computing messages through MLP
            edge_inputs = torch.cat([s_i, s_j, distance], dim = 1)  # Setup input for computing messages through MLP
        else:
            if v_i is None:
                edge_inputs = torch.cat([s_i, s_i - s_j], dim = 1)
            else:
                edge_inputs = torch.cat([s_i, s_i - s_j, v_i, v_i - v_j], dim = 1)  # Setup input for computing messages through MLP
          
        return edge_inputs
    


class OutputNetwork(torch.nn.Module):
    def __init__(self, hparams):
        super(OutputNetwork, self).__init__()
        
        if hparams["shortcut"] is "skip":
            fully_connected_dim = sum(hparams["message_dim"]) + len(hparams["scalars"]) + hparams["scalar_dim"]
        elif hparams["shortcut"] is "concat":
            fully_connected_dim = sum(hparams["message_dim"])
        else:
            fully_connected_dim = hparams["message_dim"][-1]
        
        # The graph classifier outputs a final score (without sigmoid!)
        self.network = nn.Sequential(
            make_mlp(fully_connected_dim, [hparams["final_dim"]]),
            nn.Dropout(hparams["dropout"]),
            make_mlp(hparams["final_dim"], [1], output_activation=None))
        
        
    def forward(self, global_pooled_features):
        
        return self.network(global_pooled_features)
    

class GeneralGNN(EGNNBase):
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
        self.equivariant = hparams["equivariant"]
        self.graph_construction = hparams["graph_construction"]
        self.k = hparams["k"]
        
        self.hparams["scalars"] = [] if hparams["scalars"] is None else hparams["scalars"]
        
        if (type(self.hparams["message_dim"]) is int):
            self.hparams["message_dim"] = [self.hparams["message_dim"]] * self.hparams["n_graph_iters"]
        elif (len(self.hparams["message_dim"]) == 1):
            self.hparams["message_dim"] = self.hparams["message_dim"] * self.hparams["n_graph_iters"]
        else:
            self.hparams["n_graph_iters"] = len(self.hparams["message_dim"])
        
        self.propagate_vector_dims = hparams["vector_dim"] if self.equivariant else 0
        
        self.convolution_layers = nn.ModuleList(
            [GeneralConv(
                  scalar_dim = len(self.hparams["scalars"]) + hparams["scalar_dim"], 
                  message_dim = self.hparams["message_dim"][0], 
                  invariant_vector_dim = hparams["invariant_vector_dim"], 
                  vector_dim = hparams["vector_dim"],
                  hparams = hparams)] 
            + [GeneralConv(
                  scalar_dim = self.hparams["message_dim"][i-1], 
                  message_dim = self.hparams["message_dim"][i], 
                  invariant_vector_dim = hparams["invariant_vector_dim"], 
                  vector_dim = self.propagate_vector_dims,
                  hparams = hparams)
                for i in range(1, self.hparams["n_graph_iters"])
            ])
        
        self.graph_classifier = OutputNetwork(self.hparams)
    
    def get_node_features(self, batch):
        
        # Get node vector
        v = batch[self.hparams["vector"]]
        
        # Get all scalars
        s = [batch[feature].unsqueeze(-1) 
                   if len(batch[feature].shape)==1 
                   else batch[feature] for feature in self.hparams["scalars"]]
        # Add vector invariants to the list of scalars
        s += [compute_vector_invariants(v).unsqueeze(-1)]
        
        # Handle some shaping issues
        s = torch.cat(s, dim=-1)
        if len(s.shape)==1: s=s.unsqueeze(1)
        
        return s.float(), v.float()
    
    def handle_message_passing(self, convolution_layer, batch, s_in, v_in, all_features):
        
        s_out, v_out = convolution_layer(s_in, v_in, batch.edge_index, edge_attribute = None)
        
        if self.hparams["shortcut"] is "skip":
            s_out = F.relu(torch.cat([s_out, s_in], dim=-1))
            
        elif self.hparams["shortcut"] is "concat":
            all_features.append(s_out)
        
        return s_out, v_out, all_features
    
    def get_edge_indices(self, batch):
        
        if self.graph_construction == "fully_connected":
            return batch
        elif not self.equivariant:
            batch.edge_index = knn_graph(torch.cat([batch.delta_eta.unsqueeze(-1), batch.delta_phi.unsqueeze(-1)], dim=-1), self.k, batch.batch)
        
        return batch
    
    def forward(self, batch):
        
        s, v = self.get_node_features(batch)
        
        batch = self.get_edge_indices(batch)
        
        all_features = []
        
        for i in range(self.hparams["n_graph_iters"]):                
            
            s, v, all_features = self.handle_message_passing(self.convolution_layers[i], batch, s, v, all_features)
            
            if self.graph_construction == "dynamic_knn":
                batch.edge_index = knn_graph(s, self.k, batch.batch)

        if self.hparams["shortcut"] is "concat":
            global_average = global_mean_pool(torch.cat(all_features, dim=-1), batch.batch)
        else:
            global_average = global_mean_pool(s, batch.batch)

        # Final layers
        output = self.graph_classifier(global_average)
        
        return output
    