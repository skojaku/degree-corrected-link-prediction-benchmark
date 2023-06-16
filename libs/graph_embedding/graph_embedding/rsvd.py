import numpy as np
from scipy import sparse


#
# Randomized SVD
#
def rSVD(X, dim, **params):
    if isinstance(X, list):
        return _rSVD_submatrices(X, r=dim, **params)
    else:
        return _rSVD_matrix(X, r=dim, **params)


def _rSVD_matrix(X, r, p=10, q=5):
    """Randomized SVD.

    Parameters
    ----------
    X : scipy.csr_sparse_matrix
        Matrix to decompose
    r : int
        Rank of decomposed matrix
    p : int (Optional; Default p = 5)
        Oversampling
    q : int (Optional; Default q = 1)
        Number of power iterations

    Return
    ------
    U : numpy.ndrray
        Left singular vectors of size (X.shape[0], r)
    lams : numpy.ndarray
        Singular values of size (r,)
    V : numpy.ndarray
        Right singular vectors of size (X.shape[0], r)
    """
    Nr, Nc = X.shape
    dim = r + p
    R = np.random.randn(Nc, dim)
    Z = X @ R
    for _i in range(q):
        Z = X @ (X.T @ Z)
    Q, R = np.linalg.qr(Z, mode="reduced")
    Y = Q.T @ X
    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    selected = np.argsort(np.abs(S))[::-1][0:r]
    return U[:, selected], S[selected], VT[selected, :]


def _rSVD_submatrices(mat_seq, r, p=10, q=5, fill_zero=1e-20):
    """Randomized SVD for decomposable matrix. We assume that the matrix is
    given by mat_seq[0] + mat_seq[1],...

    Parameters
    ----------
    mat_seq: list
        List of decomposed matrices
    r : int
        Rank of decomposed matrix
    p : int (Optional; Default p = 10)
        Oversampling
    q : int (Optional; Default q = 1)
        Number of power iterations
    fill_zero: float
        Replace the zero values in the transition matrix with this value.

    Return
    ------
    U : numpy.ndrray
        Left singular vectors of size (X.shape[0], r)
    lams : numpy.ndarray
        Singular values of size (r,)
    V : numpy.ndarray
        Right singular vectors of size (X.shape[0], r)
    """
    Nc = mat_seq[-1][-1].shape[1]
    dim = r + p

    R = np.random.randn(Nc, dim)  # Random gaussian matrix
    Z = mat_prod_matrix_seq(mat_seq, R)

    for _i in range(q):  # Power iterations
        zz = mat_prod_matrix_seq(Z.T, mat_seq)
        Z = mat_prod_matrix_seq(mat_seq, zz.T)
    Q, R = np.linalg.qr(Z, mode="reduced")

    Y = mat_prod_matrix_seq(Q.T, mat_seq)

    UY, S, VT = np.linalg.svd(Y, full_matrices=0)
    U = Q @ UY
    selected = np.argsort(np.abs(S))[::-1][0:r]

    U, S, VT = U[:, selected], S[selected], VT[selected, :]
    if isinstance(U, np.matrix):
        U = np.asarray(U)
    if isinstance(VT, np.matrix):
        VT = np.asarray(VT)
    if isinstance(S, np.matrix):
        S = np.asarray(S).reshape(-1)
    return U, S, VT


def assemble_matrix_from_list(matrix_seq):
    S = None
    for k in range(len(matrix_seq)):
        R = matrix_seq[k][0]
        for i in range(1, len(matrix_seq[k])):
            R = R @ matrix_seq[k][i]

        if sparse.issparse(R):
            R = R.toarray()

        if S is None:
            S = R
        else:
            S += R
    return S


def mat_prod_matrix_seq(A, B):
    def right_mat_prod_matrix_seq(A, matrix_seq):

        S = None
        for k in range(len(matrix_seq)):
            R = A @ matrix_seq[k][0]
            if sparse.issparse(R):
                R = R.toarray()

            for rid in range(1, len(matrix_seq[k])):
                R = R @ matrix_seq[k][rid]

            if S is None:
                S = R
            else:
                S = S + R
        return S

    def left_mat_prod_matrix_seq(matrix_seq, A):

        S = None
        for k in range(len(matrix_seq)):
            R = matrix_seq[k][-1] @ A
            if sparse.issparse(R):
                R = R.toarray()
            for rid in range(1, len(matrix_seq[k])):
                R = matrix_seq[k][-rid - 1] @ R

            if S is None:
                S = R
            else:
                S = S + R
        return S

    if isinstance(A, list) and not isinstance(B, list):
        return left_mat_prod_matrix_seq(A, B)
    elif isinstance(B, list) and not isinstance(A, list):
        return right_mat_prod_matrix_seq(A, B)


def mat_seq_transpose(matrix_seq):
    retval = []
    for _i, seq in enumerate(matrix_seq):
        _seq = []
        for _j, s in enumerate(seq[::-1]):
            _seq += [s.T]
        retval += [_seq]
    return retval
