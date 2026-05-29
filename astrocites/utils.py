import numpy as np


def bernoulli_loader(data, time=None, dt=1.0, **kwargs):
    for i in range(len(data)):
        yield data[i]


def create_adjacency_matrix(N):
    total_cells = N * N
    adjacency_matrix = np.zeros((total_cells, total_cells), dtype=int)
    for i in range(total_cells):
        row = i // N
        col = i % N
        if col > 0:
            j = i - 1
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        if col < N - 1:
            j = i + 1
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        if row > 0:
            j = i - N
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
        if row < N - 1:
            j = i + N
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
    return adjacency_matrix