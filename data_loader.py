import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


def get_four_momenta(jet_tuple):
    energies = np.array([getattr(jet_tuple, f'E_{i}') for i in range(200)])
    x_values = np.array([getattr(jet_tuple, f'PX_{i}') for i in range(200)])
    y_values = np.array([getattr(jet_tuple, f'PY_{i}') for i in range(200)])
    z_values = np.array([getattr(jet_tuple, f'PZ_{i}') for i in range(200)])

    existing_jet_mask = energies > 0
    energies, x_values, y_values, z_values = energies[existing_jet_mask], x_values[existing_jet_mask], y_values[
        existing_jet_mask], z_values[existing_jet_mask]

    p = torch.from_numpy(np.stack([energies, x_values, y_values, z_values])).T
    return p


def build_dataset(dataframe, num_jets = None):
    momenta, truth_labels = [], []
    if num_jets is not None:
        subsample = dataframe.sample(n = num_jets)
    else:
        subsample = dataframe

    for jet in subsample.itertuples():
        p = get_four_momenta(jet)
        y = torch.tensor(jet.is_signal_new)

        momenta.append(p)
        truth_labels.append(y)

    return momenta, truth_labels


class JetDataset(Dataset):
    def __init__(self, p, y):
        super(JetDataset).__init__()
        self.p = p
        self.y = y

    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        this_p = self.p[idx].float()
        this_y = self.y[idx].long()

        sample = {"p": this_p, "y": this_y}
        return sample


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
    return torch.tensor(edges)


if __name__ == '__main__':
    test_file = 'test.h5'
    with pd.HDFStore(test_file, mode = 'r') as store:
        test_df = store['table']

    all_p, all_y = build_dataset(test_df, 1000)
    train_dataset = JetDataset(all_p, all_y)
    train_loader = DataLoader(train_dataset)
    print(all_p)
