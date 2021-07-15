from random import random
from math import cos, sin, pi
from copy import deepcopy
import torch
from lorentz_equivariant_gnn.legnn_model import LEGNN, get_edges_batch


def rotate_x(lorentz_vector, theta):
    c, s = cos(theta), sin(theta)
    rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, c, -s],
                                    [0, 0, s, c]])
    return (rotation_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def rotate_y(lorentz_vector, theta):
    c, s = cos(theta), sin(theta)
    rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, c, 0, s],
                                    [0, 0, 1, 0],
                                    [0, -s, 0, c]])
    return (rotation_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def rotate_z(lorentz_vector, theta):
    c, s = cos(theta), sin(theta)
    rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, c, -s, 0],
                                    [0, s, c, 0],
                                    [0, 0, 0, 1]])
    return (rotation_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


if __name__ == '__main__':
    test_angle = 2 * pi * random()

    # Dummy parameters
    batch_size = 1  # 8
    n_nodes = 4
    n_feat = 1
    x_dim = 4

    # Dummy variables h, x and fully connected edges
    h = torch.rand(batch_size * n_nodes, n_feat)
    x1 = torch.rand(batch_size * n_nodes, x_dim)
    x2 = deepcopy(x1)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)

    print("Input: " + str(x1))

    # Initialize LEGNN
    legnn = LEGNN(input_feature_dim = n_feat, message_dim = 32, output_feature_dim = 1, edge_feature_dim = 1)

    # Run LEGNN

    # First rotate the lorentz vector before passing through network
    x1 = rotate_x(x1, test_angle)
    h, x1 = legnn(h, x1, edges, edge_attr)
    print(x1)

    # Now rotate the lorentz vector after passing through the network
    h, x2 = legnn(h, x2, edges, edge_attr)
    x2 = rotate_x(x2, test_angle)
    print(x2)

    print(torch.isclose(x1, x2, atol = 1e-8, rtol = 1e-3))