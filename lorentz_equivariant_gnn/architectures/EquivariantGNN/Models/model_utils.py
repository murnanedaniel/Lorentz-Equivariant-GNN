import torch

def compute_radials(edge_index, x):
    """
    Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j

    :param edge_index: Array containing the connection between nodes
    :param x: The coordinates (node embeddings)
    :return: Minkowski distances (squared) and coordinate differences x_i - x_j
    """

    row, col = edge_index
    coordinate_differences = x[row] - x[col]
    minkowski_distance_squared = coordinate_differences ** 2
    minkowski_distance_squared[:, 0] = -minkowski_distance_squared[:, 0]  # Place minus sign on time coordinate as \eta = diag(-1, 1, 1, 1)
    radial = torch.sum(minkowski_distance_squared, 1).unsqueeze(1)
    return radial, coordinate_differences

def compute_initial_feature(x):
    """
    Calculates the Minkowski distance (squared) between coordinates (node embeddings) x_i and x_j

    :param edge_index: Array containing the connection between nodes
    :param x: The coordinates (node embeddings)
    :return: Minkowski distances (squared) and coordinate differences x_i - x_j
    """

    momentum_squared = x**2
    momentum_squared[:, 0] = -momentum_squared[:, 0]
    minkowski_magnitude = torch.sum(momentum_squared, 1)

    return minkowski_magnitude