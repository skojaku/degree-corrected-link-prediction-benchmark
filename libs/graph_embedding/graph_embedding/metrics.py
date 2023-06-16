import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from scipy import sparse

def eval_separation_lda(emb, group_ids):
    def ind2mat(arr, Nr = None, Nc = None):
        if Nr is None:
            Nr = len(arr)
        if Nc is None:
            Nc = int(np.max(arr) + 1)
        return sparse.csr_matrix((np.ones_like(arr), (np.arange(Nr), arr.astype(int))), shape=(Nr, Nc))

    gids, _ = divmod(np.arange(emb.shape[0]), (emb.shape[0] / 2))
    gids = gids.astype(int)

    clf = LinearDiscriminantAnalysis()
    x = clf.fit_transform(emb, gids)
    score = roc_auc_score(gids, x)
    if score < 0.5:
        score = 1 - score
    return score