import numpy as np


def evaluate(model, truth, users, seen, k=10):
    recall, ndcg = [], []
    discounts = 1 / np.log2(np.arange(2, k + 2))

    for user in users:
        scores = model.score([user])[0]
        scores[seen[user].indices] = -np.inf
        recommendations = np.argsort(-scores, kind="stable")[:k]

        relevant = truth[user].indices
        hits = np.isin(recommendations, relevant)
        recall.append(hits.sum() / len(relevant))
        ideal = discounts[:min(len(relevant), k)].sum()
        ndcg.append((hits * discounts).sum() / ideal)

    return {f"Recall@{k}": float(np.mean(recall)), f"NDCG@{k}": float(np.mean(ndcg))}
