import numpy as np


class Oracle:

    def __init__(self, weights, capacity):

        self.weights = weights
        self.capacity = capacity

        self.n_queries = 0

    def query(self, point: np.ndarray):

        self.n_queries += 1

        return int(np.dot(self.weights, point) <= self.capacity)
