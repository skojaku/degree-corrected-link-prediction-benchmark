# %%
import numba
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


class PairSampler:
    """Sample sample pair samplers conditioned on class labels.

    The sampler samples positive and negative pairs of samples. A positive sample
    pair is a pair of samples with the same class labels, i.e.,
    `labels[i] = labels[j]`.

    Argument `negative_labels` enables a conditional sampling, i.e., a negative
    pair (i,j) is sampled such that `labels[i] != labels[j]` and
    `negative_labels[i] = negative_labels[j]`.

    Examples:
        ```python
        group_ids = np.kron(np.arange(3), np.ones(10))
        block_ids = np.kron(np.arange(2), np.ones(15))
        sampler = PairSampler(labels=group_ids, negative_labels=block_ids)
        sampler.sample_positive_pairs(10)
        sampler.sample_negative_pairs(10)
        ```
    """

    def __init__(self, labels, negative_labels=None):

        if negative_labels is None:
            negative_labels = np.zeros_like(labels)

        labels = np.unique(labels, return_inverse=True)[1]
        negative_labels = np.unique(negative_labels, return_inverse=True)[1]

        #
        # Generate a new index concatenating the group and block ids
        #
        pos_neg_label_ids = np.unique(
            pairing(negative_labels, labels, unordered=False), return_inverse=True
        )[1].astype(int)

        # Prep matrix for fast sampling
        Ubg2b = make_member_matrix(
            row_ids=pos_neg_label_ids,
            col_ids=negative_labels,
            is_ids=True,
            binarize=True,
        )
        Ubg2g = make_member_matrix(
            row_ids=pos_neg_label_ids, col_ids=labels, is_ids=True, binarize=True
        )
        Usample = toMemberMetrix(pos_neg_label_ids)

        Ag, Ab = Ubg2g @ Ubg2g.T, Ubg2b @ Ubg2b.T
        Abg = Ab - Ag
        Abg.data = np.maximum(0, Abg.data)
        Abg.eliminate_zeros()
        PosGroup2sample, NegGroup2sample = (
            Ag @ sparse.csr_matrix(Usample.T),
            Abg @ sparse.csr_matrix(Usample.T),
        )

        PosGroup2sample.eliminate_zeros()
        NegGroup2sample.eliminate_zeros()
        PosGroup2sample, NegGroup2sample = (
            cum_trans_prob(PosGroup2sample),
            cum_trans_prob(NegGroup2sample),
        )  # preprocess

        self.PosGroup2sample = PosGroup2sample
        self.NegGroup2sample = NegGroup2sample
        self.labels = labels
        self.negative_labels = negative_labels
        self.pos_neg_label_ids = pos_neg_label_ids
        self.num_group = Ubg2g.shape[1]
        self.num_block = Ubg2b.shape[1]
        self.num_group_block = Ubg2b.shape[0]

    def sample_anchor_positive_negative_triplet(self, num_samples):
        # Random sample positive node pairs
        num_sampled = 0
        anc_sampled, pos_sampled, neg_sampled = [], [], []
        pbar = tqdm(total=num_samples)
        while num_sampled < num_samples:
            rpos_group_block = np.random.choice(
                self.num_group_block, num_samples - num_sampled, replace=True
            )
            anc = sample_columns_from_cum_prob(
                rpos_group_block, self.PosGroup2sample, preprocessed=True
            )
            pos = sample_columns_from_cum_prob(
                rpos_group_block, self.PosGroup2sample, preprocessed=True
            )
            neg = sample_columns_from_cum_prob(
                rpos_group_block, self.NegGroup2sample, preprocessed=True
            )
            s = (anc >= 0) * (pos >= 0) * (neg >= 0) * (anc != pos)
            anc, pos, neg = anc[s], pos[s], neg[s]

            pbar.update(len(pos))

            anc_sampled.append(anc)
            pos_sampled.append(pos)
            neg_sampled.append(neg)
            num_sampled += len(pos)

        anc, pos, neg = (
            np.concatenate(anc_sampled),
            np.concatenate(pos_sampled),
            np.concatenate(neg_sampled),
        )
        return anc, pos, neg

    def sample_positive_pairs(self, num_samples):
        # Random sample positive node pairs
        num_sampled = 0
        rpos_sampled, cpos_sampled = [], []
        while num_sampled < num_samples:
            rpos_group_block = np.random.choice(
                self.num_group_block, num_samples - num_sampled, replace=True
            )
            rpos = sample_columns_from_cum_prob(
                rpos_group_block, self.PosGroup2sample, preprocessed=True
            )
            cpos = sample_columns_from_cum_prob(
                rpos_group_block, self.PosGroup2sample, preprocessed=True
            )
            s = (rpos >= 0) * (cpos >= 0) * (rpos != cpos)
            rpos, cpos = rpos[s], cpos[s]

            rpos_sampled.append(rpos)
            cpos_sampled.append(cpos)
            num_sampled += len(rpos)
        rpos, cpos = np.concatenate(rpos_sampled), np.concatenate(cpos_sampled)
        return rpos, cpos

    def sample_negative_pairs(self, num_samples):
        # Random sample positive node pairs
        num_sampled = 0
        rneg_sampled, cneg_sampled = [], []
        while num_sampled < num_samples:
            rneg_group_block = np.random.choice(
                self.num_group_block, num_samples - num_sampled, replace=True
            )
            rneg = sample_columns_from_cum_prob(
                rneg_group_block, self.PosGroup2sample, preprocessed=True
            )
            cneg = sample_columns_from_cum_prob(
                rneg_group_block, self.NegGroup2sample, preprocessed=True
            )
            s = (rneg >= 0) * (cneg >= 0) * (rneg != cneg)
            rneg, cneg = rneg[s], cneg[s]

            rneg_sampled.append(rneg)
            cneg_sampled.append(cneg)
            num_sampled += len(rneg)
        rneg, cneg = np.concatenate(rneg_sampled), np.concatenate(cneg_sampled)
        return rneg, cneg


def cum_trans_prob(A):
    def _calc_cum_trans_prob(
        A_indptr, A_indices, A_data_, num_nodes,
    ):
        A_data = A_data_.copy()
        for i in range(num_nodes):
            # Compute the out-deg
            outdeg = np.sum(A_data[A_indptr[i] : A_indptr[i + 1]])
            A_data[A_indptr[i] : A_indptr[i + 1]] = np.cumsum(
                A_data[A_indptr[i] : A_indptr[i + 1]]
            ) / np.maximum(outdeg, 1)
        return A_data

    P = A.copy()
    a = _calc_cum_trans_prob(P.indptr, P.indices, P.data.astype(float), P.shape[0])
    P.data = a
    return P


def sample_columns_from_cum_prob(rows, A, preprocessed=False):
    if preprocessed is False:
        A = cum_trans_prob(A)
    return _sample_columns_from_cum_prob(rows, A.indptr, A.indices, A.data)


@numba.jit(nopython=True, parallel=False)
def _sample_columns_from_cum_prob(rows, A_indptr, A_indices, A_data):
    retvals = -np.ones(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        r = rows[i]
        nnz_row = A_indptr[r + 1] - A_indptr[r]
        if nnz_row == 0:
            continue

        # find a neighbor by a roulette selection
        _ind = np.searchsorted(
            A_data[A_indptr[r] : A_indptr[r + 1]], np.random.rand(), side="right",
        )
        retvals[i] = A_indices[A_indptr[r] + _ind]
    return retvals


def pairing(k1, k2, unordered=False):
    """Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function."""
    k12 = k1 + k2
    if unordered:
        return (k12 * (k12 + 1)) * 0.5 + np.minimum(k1, k2)
    else:
        return (k12 * (k12 + 1)) * 0.5 + k2


def depairing(z):
    """Inverse of Cantor pairing function http://en.wikipedia.org/wiki/Pairing_
    function#Inverting_the_Cantor_pairing_function."""
    w = np.floor((np.sqrt(8 * z + 1) - 1) * 0.5)
    t = (w ** 2 + w) * 0.5
    y = np.round(z - t).astype(np.int64)
    x = np.round(w - y).astype(np.int64)
    return x, y


def make_member_matrix(
    row_ids, col_ids, shape=None, w=None, is_ids=False, binarize=False
):
    if w is None:
        w = np.ones(len(row_ids))

    if is_ids:
        if shape is None:
            shape = (np.max(row_ids) + 1, np.max(col_ids) + 1)
        B = sparse.csr_matrix((w, (row_ids, col_ids)), shape=shape)

        if binarize:
            B.data = B.data * 0 + 1
        return B
    else:
        row_list, r = np.unique(row_ids, return_inverse=True)
        col_list, c = np.unique(col_ids, return_inverse=True)
        if shape is None:
            shape = (len(row_list), len(col_list))
        B = sparse.csr_matrix((w, (r, c)), shape=shape)
        if binarize:
            B.data = B.data * 0 + 1
        return B, row_list, col_list


def toMemberMetrix(ids, shape=None, is_numeric=True):
    if is_numeric:
        if shape is None:
            shape = (len(ids), int(np.max(ids) + 1))
        return sparse.csr_matrix(
            (np.ones_like(ids), (np.arange(len(ids)), ids)), shape=shape
        )
    else:
        ids = np.unique(ids, return_inverse=True)[1]
        if shape is None:
            shape = (len(ids), int(np.max(ids) + 1))
        return sparse.csr_matrix(
            (np.ones_like(ids), (np.arange(len(ids)), ids)), shape=shape
        )
