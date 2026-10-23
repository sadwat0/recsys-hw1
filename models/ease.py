import numpy as np
from scipy.linalg import solve


class EASE:
    def __init__(self, reg=100):
        self.reg = reg

    def fit(self, counts):
        self.history = counts.astype(np.float64)
        self.history.data[:] = 1

        gram = (self.history.T @ self.history).toarray()
        gram += self.reg * np.eye(gram.shape[0])
        inverse = solve(gram, np.eye(gram.shape[0]), assume_a="pos", check_finite=False)

        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0)
        self.weights = self.weights.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray(self.history[users] @ self.weights, dtype=np.float32)
