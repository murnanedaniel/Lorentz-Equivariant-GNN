import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


def calc_eta_phi(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(r, z)
    eta = -1. * np.log(np.tan(theta / 2.))
    phi = np.arctan2(y, x)

    return eta, phi


def get_eta_phi_values(jet_tuple):
    energies = np.array([getattr(jet_tuple, f'E_{i}') for i in range(200)])
    x_values = np.array([getattr(jet_tuple, f'PX_{i}') for i in range(200)])
    y_values = np.array([getattr(jet_tuple, f'PY_{i}') for i in range(200)])
    z_values = np.array([getattr(jet_tuple, f'PZ_{i}') for i in range(200)])

    existing_jet_mask = energies > 0
    energies, x_values, y_values, z_values = energies[existing_jet_mask], x_values[existing_jet_mask], y_values[
        existing_jet_mask], z_values[existing_jet_mask]

    eta_values, phi_values = calc_eta_phi(x_values, y_values, z_values)

    X = torch.from_numpy(np.stack([x_values, y_values, z_values, eta_values, phi_values])).T
    energies = torch.from_numpy(energies)

    return X, energies


def build_dataset(dataframe, num_jets = None):
    all_X, all_energies, all_y = [], [], []
    if num_jets is not None:
        subsample = dataframe.sample(n = num_jets)
    else:
        subsample = dataframe

    for jet in subsample.itertuples():
        X, energies = get_eta_phi_values(jet)
        y = torch.tensor(jet.is_signal_new)

        all_X.append(X)
        all_energies.append(energies)
        all_y.append(y)

    return all_X, all_energies, all_y


class JetDataset(Dataset):
    def __init__(self, X, E, y):
        super(JetDataset).__init__()

        self.X = X
        self.E = E
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        this_X = self.X[idx].float()
        this_E = self.E[idx].float()
        this_y = self.y[idx].long()

        sample = {"X": this_X, "E": this_E, "y": this_y}

        return sample


def extract_four_momenta(data):
    four_momenta = []
    for jet in data:
        #four_momenta.append([])
        locality_data = jet["X"]
        energy_data = jet["E"]
        for i in range(len(locality_data)):
            if i == 0:
                four_momenta.append(torch.cat((energy_data[i].view(1), locality_data[i, :-2])))
            elif i == 1:
                four_momenta[-1] = torch.stack((four_momenta[-1], torch.cat((energy_data[i].view(1), locality_data[i, :-2]))))
            else:
                four_momenta[-1] = torch.cat((four_momenta[-1], torch.unsqueeze(torch.cat((energy_data[i].view(1), locality_data[i, :-2])), dim = 0)))
            #four_momenta[-1].append(torch.cat((energy_data[i].view(1), locality_data[i, :-2])))

    return four_momenta


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

    all_X, all_energies, all_y = build_dataset(test_df, 1000)
    train_dataset = JetDataset(all_X, all_energies, all_y)
    train_loader = DataLoader(train_dataset)

    #print(train_dataset[0])
    #print(train_dataset[0]["X"])

    print(extract_four_momenta([train_dataset[0]]))

    edges = get_edges(len(train_dataset[0]["X"]))

    plt.figure(figsize=(10, 10))
    for edge in edges.T:
        plt.plot(train_dataset[0]["X"][edge, -2], train_dataset[0]["X"][edge, -1], c="k", linewidth=1)

    plt.scatter(train_dataset[0]["X"][:, -2], train_dataset[0]["X"][:, -1], s=100)
    plt.show()
