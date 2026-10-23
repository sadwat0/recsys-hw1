import numpy as np
from scipy.sparse import csr_matrix


class ItemKNN:
    def __init__(self, k=50):
        self.k = k

    def fit(self, counts):
        self.history = counts.copy().astype(np.float32)
        self.history.data[:] = 1

        norms = np.sqrt(np.asarray(self.history.sum(axis=0)).ravel())
        similarity = (self.history.T @ self.history).toarray()
        similarity /= np.maximum(norms[:, None] * norms[None, :], 1)
        np.fill_diagonal(similarity, 0)

        for item in range(similarity.shape[1]):
            order = np.argsort(-similarity[:, item], kind="stable")
            similarity[order[self.k:], item] = 0

        self.weights = csr_matrix(similarity)
        return self

    def score(self, users):
        return (self.history[users] @ self.weights).toarray()
