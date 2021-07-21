import torch

from lorentz_equivariant_gnn.legnn_model import LEGNN, get_edges_batch
from data_loader import *


train_file = '../test.h5'#'../train.h5'
with pd.HDFStore(train_file, mode = 'r') as store:
    train_df = store['table']

all_X, all_energies, all_y = build_dataset(train_df, 1000)
train_dataset = JetDataset(all_X, all_energies, all_y)
train_loader = DataLoader(train_dataset)

four_momenta = extract_four_momenta(train_dataset)
input_momenta_count = list(four_momenta[0].size())[0]

h = torch.ones(input_momenta_count, 1)
x = four_momenta[0]
edges, edge_attr = get_edges_batch(input_momenta_count, 1)

print("Input:\n" + str(x))

legnn_network = LEGNN(input_feature_dim = 1, message_dim = 32, output_feature_dim = 1, edge_feature_dim = 1)

h, x = legnn_network(h, x, edges, edge_attr)
print("Output:\n" + str(x))
