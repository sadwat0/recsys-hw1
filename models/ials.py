import numpy as np
from scipy.linalg import solve


class IALS:
    def __init__(self, factors=32, alpha=10, reg=10, iterations=8, seed=42):
        self.factors = factors
        self.alpha = alpha
        self.reg = reg
        self.iterations = iterations
        self.seed = seed

    def update(self, counts, target, other):
        result = np.zeros((counts.shape[0], self.factors), dtype=np.float32)
        base = other.T @ other + self.reg * np.eye(self.factors, dtype=np.float32)
        for row in range(counts.shape[0]):
            start, end = counts.indptr[row:row + 2]
            vectors = other[counts.indices[start:end]]
            confidence = 1 + self.alpha * np.log1p(counts.data[start:end])
            matrix = base + (vectors.T * (confidence - 1)) @ vectors
            rhs = vectors.T @ (confidence * target.data[start:end])
            result[row] = solve(matrix, rhs, assume_a="pos", check_finite=False)
        return result

    def fit(self, counts, target=None):
        counts = counts.astype(np.float32).tocsr()
        counts.sort_indices()
        if target is None:
            target = counts.copy()
            target.data[:] = 1
        target = target.astype(np.float32).tocsr()
        target.sort_indices()
        assert np.array_equal(counts.indptr, target.indptr)
        assert np.array_equal(counts.indices, target.indices)
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.01, (counts.shape[1], self.factors)).astype(np.float32)
        reverse_counts = counts.T.tocsr()
        reverse_target = target.T.tocsr()
        for _ in range(self.iterations):
            self.users = self.update(counts, target, self.items)
            self.items = self.update(reverse_counts, reverse_target, self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
