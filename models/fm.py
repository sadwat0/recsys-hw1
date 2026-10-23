import numpy as np
import torch
from torch import nn


class FM(nn.Module):
    def __init__(self, factors=16, lr=0.01, reg=1e-5, epochs=6, seed=42):
        super().__init__()
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.seed = seed

    def forward(self, features):
        vectors = self.embedding(features)
        interaction = 0.5 * (vectors.sum(dim=1).square() - vectors.square().sum(dim=1))
        return self.bias + self.linear(features).sum(dim=(1, 2)) + interaction.sum(dim=1)

    def fit(self, counts, item_features, seen):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.n_users, self.n_items = counts.shape
        self.item_features = torch.tensor(item_features + self.n_users, dtype=torch.long)
        size = int(self.item_features.max()) + 1
        self.embedding = nn.Embedding(size, self.factors)
        self.linear = nn.Embedding(size, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.zeros_(self.linear.weight)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        positive_users, positive_items = counts.nonzero()
        known = seen.toarray() > 0
        assert (~known).any(axis=1).all()

        for _ in range(self.epochs):
            negative_items = rng.integers(self.n_items, size=len(positive_users))
            rejected = known[positive_users, negative_items]
            while rejected.any():
                negative_items[rejected] = rng.integers(self.n_items, size=rejected.sum())
                rejected = known[positive_users, negative_items]

            users = np.concatenate([positive_users, positive_users])
            items = np.concatenate([positive_items, negative_items])
            labels = np.concatenate([np.ones(len(positive_users)), np.zeros(len(positive_users))])
            order = rng.permutation(len(users))

            for start in range(0, len(order), 8192):
                indices = order[start:start + 8192]
                user_ids = torch.tensor(users[indices, None], dtype=torch.long)
                features = torch.cat([user_ids, self.item_features[items[indices]]], dim=1)
                target = torch.tensor(labels[indices], dtype=torch.float32)

                logits = self(features)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
                penalty = self.embedding(features).square().mean() + self.linear(features).square().mean()
                loss = loss + self.reg * penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    @torch.no_grad()
    def score(self, users):
        result = []
        for user in users:
            user_ids = torch.full((self.n_items, 1), int(user), dtype=torch.long)
            features = torch.cat([user_ids, self.item_features], dim=1)
            result.append(self(features).numpy())
        return np.asarray(result)
