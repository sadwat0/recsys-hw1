import numpy as np

from .ials import IALS


class CustomIALS(IALS):
    def fit(self, counts, pairs):
        fraction = pairs["fraction"].to_numpy()
        repeats = np.minimum(np.log1p(pairs["count"].to_numpy()) / np.log(6), 1)
        interest = np.clip(
            0.5 * fraction + 0.25 * repeats
            + 0.25 * pairs["liked"].to_numpy() - 0.5 * pairs["disliked"].to_numpy(),
            0, 1,
        )

        target = counts.copy()
        target.data = interest.astype(np.float32)
        return super().fit(counts, target)
