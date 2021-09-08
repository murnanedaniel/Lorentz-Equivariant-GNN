import sys
import os
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data
from torch import nn


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=False
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


def load_datasets(input_dir, data_split):
    
    train_file = os.path.join(input_dir, 'test.h5')
    with pd.HDFStore(train_file, mode = 'r') as store:
        train_df = store['table']

    val_file = os.path.join(input_dir, 'val.h5')
    with pd.HDFStore(train_file, mode = 'r') as store:
        val_df = store['table']

    train_dataset = build_dataset(train_df, data_split[0])
    val_dataset = build_dataset(val_df, data_split[1])
     
    return train_dataset, val_dataset



def get_four_momenta(jet_tuple):
    energies = np.array([getattr(jet_tuple, f'E_{i}') for i in range(200)])
    x_values = np.array([getattr(jet_tuple, f'PX_{i}') for i in range(200)])
    y_values = np.array([getattr(jet_tuple, f'PY_{i}') for i in range(200)])
    z_values = np.array([getattr(jet_tuple, f'PZ_{i}') for i in range(200)])

    existing_jet_mask = True#energies > 0
    energies, x_values, y_values, z_values = energies[existing_jet_mask], x_values[existing_jet_mask], y_values[
        existing_jet_mask], z_values[existing_jet_mask]

    p = torch.from_numpy(np.stack([energies, x_values, y_values, z_values])).T.squeeze()
    return p


def build_dataset(dataframe, num_jets = None):
    
    dataset = []
    
    if num_jets is not None:
        subsample = dataframe.sample(n = num_jets)
    else:
        subsample = dataframe

    for jet in subsample.itertuples():
        p = get_four_momenta(jet)
        y = torch.tensor(jet.is_signal_new)
        e = get_edges(len(p))

        dataset.append(Data(x=p, y=y, edge_index=e))

    return dataset


# class JetDataset(Dataset):
#     def __init__(self, p, y):
#         super(JetDataset).__init__()
#         self.p = p
#         self.y = y

#     def __len__(self):
#         return len(self.p)

#     def __getitem__(self, idx):
#         this_p = self.p[idx].float()
#         this_y = self.y[idx].long()

#         sample = {"p": this_p, "y": this_y}
#         return sample


"""
Returns an array of edge links corresponding to a fully-connected graph - OLD VERSION
"""
# def get_edges(n_nodes):
#     rows, cols = [], []
#     for i in range(n_nodes):
#         for j in range(n_nodes):
#             if i != j:
#                 rows.append(i)
#                 cols.append(j)

#     edges = [rows, cols]
#     return torch.tensor(edges)

"""
Returns an array of edge links corresponding to a fully-connected graph - NEW VERSION
"""
def get_edges(n_nodes):
    
    node_list = torch.arange(n_nodes)
    edges = torch.combinations(node_list, r=2).T
    bidirectional_edges = torch.cat([edges, edges.flip(0)], axis=1)
    
    return bidirectional_edges

if __name__ == '__main__':
    test_file = 'test.h5'
    with pd.HDFStore(test_file, mode = 'r') as store:
        test_df = store['table']

    all_p, all_y = build_dataset(test_df, 1000)
    train_dataset = JetDataset(all_p, all_y)
    train_loader = DataLoader(train_dataset)
    print(all_p)