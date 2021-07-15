from random import random
from math import cos, sin, pi, sqrt
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


def boost_x(lorentz_vector, gamma):
    beta = sqrt(1 - 1 / (gamma ** 2))
    boost_matrix = torch.tensor([[gamma, -beta * gamma, 0, 0],
                                 [-beta * gamma, gamma, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
    return (boost_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def boost_y(lorentz_vector, gamma):
    beta = sqrt(1 - 1 / (gamma ** 2))
    boost_matrix = torch.tensor([[gamma, 0, -beta * gamma, 0],
                                 [0, 1, 0, 0],
                                 [-beta * gamma, 0, gamma, 0],
                                 [0, 0, 0, 1]])
    return (boost_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def boost_z(lorentz_vector, gamma):
    beta = sqrt(1 - 1 / (gamma ** 2))
    boost_matrix = torch.tensor([[gamma, 0, 0, -beta * gamma],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [-beta * gamma, 0, 0, gamma]])
    return (boost_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def rotate(lorentz_vector, theta_x, theta_y, theta_z):
    # Rotate in the order of x, y, and z
    return rotate_z(rotate_y(rotate_x(lorentz_vector, theta_x), theta_y), theta_z)


def boost(lorentz_vector, gamma_x, gamma_y, gamma_z):
    # Rotate in the order of x, y, and z
    return boost_z(boost_y(boost_x(lorentz_vector, gamma_x), gamma_y), gamma_z)


if __name__ == '__main__':
    test_angle_x = 2 * pi * random()
    test_angle_y = 2 * pi * random()
    test_angle_z = 2 * pi * random()
    test_boost_x = 2 * random() + 1
    test_boost_y = 2 * random() + 1
    test_boost_z = 2 * random() + 1

    print(test_boost_x)

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
    legnn = LEGNN(input_feature_dim = n_feat, message_dim = 32, output_feature_dim = 1, edge_feature_dim = 1, n_layers = 4)

    # Run LEGNN

    # First rotate the lorentz vector before passing through network
    x1 = boost(rotate(x1, test_angle_x, test_angle_y, test_angle_z), test_boost_x, test_boost_y, test_boost_z)
    h, x1 = legnn(h, x1, edges, edge_attr)
    print(x1)

    # Now rotate the lorentz vector after passing through the network
    h, x2 = legnn(h, x2, edges, edge_attr)
    x2 = boost(rotate(x2, test_angle_x, test_angle_y, test_angle_z), test_boost_x, test_boost_y, test_boost_z)
    print(x2)

    print(torch.isclose(x1, x2, atol = 1e-8, rtol = 1e-3))