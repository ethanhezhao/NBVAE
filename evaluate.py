import numpy as np
import bottleneck as bn


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):

    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def precisionAtK(Y_pred_orig, Y_true_orig, k, verbose=False):
    Y_pred = Y_pred_orig.copy()
    Y_true = Y_true_orig.copy()
    row_sum = np.asarray(Y_true.sum(axis=1)).reshape(-1)
    indices = row_sum.argsort()
    row_sum.sort()
    start = 0
    while start < len(indices) and row_sum[start] == 0:
        start += 1
    indices = indices[start:]
    Y_pred = Y_pred[indices, :]
    Y_true = Y_true[indices, :]
    p = np.zeros(k)
    assert Y_pred.shape == Y_true.shape
    n_items, n_labels = Y_pred.shape
    prevMatch = 0
    for i in range(1, k + 1):
        Jidx = np.argmax(Y_pred, 1)
        prevMatch += np.sum(Y_true[np.arange(n_items), Jidx])
        Y_pred[np.arange(n_items), Jidx] = -np.inf
        p[i - 1] = prevMatch / (i * n_items)
    return tuple(p[[0, 2, 4]])

def evaluate_all(X_pred, X_true):

    ranks = [1, 5, 10, 20, 50]
    ndcgs = np.zeros((len(ranks), X_pred.shape[0]))
    recalls = np.zeros((len(ranks), X_pred.shape[0]))
    for i in range(len(ranks)):
        ndcgs[i, :] = NDCG_binary_at_k_batch(X_pred, X_true, ranks[i])
        recalls[i, :] = Recall_at_k_batch(X_pred, X_true, ranks[i])
    return ndcgs, recalls

    return result