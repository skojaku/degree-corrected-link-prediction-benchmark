import numpy as np 
from scipy import sparse 
from scipy import sparse, stats
from sklearn.decomposition import TruncatedSVD

A = sparse.load_npz("../../../data/multi_partition_model/networks/net_n~1000_K~2_cave~10_mu~0.20_sample~0.npz")


deg = np.array(A.sum(axis = 1)).reshape(-1)
N = A.shape[0]
Z = sparse.csr_matrix((N, N))
I = sparse.identity(N, format="csr")
D = sparse.diags(deg)
denom = 1 / (deg -1)
denom[np.isnan(denom)] = 0
denom[np.isinf(denom)] = 0
DIinv = sparse.diags(denom)

B = sparse.bmat([[Z, D - I], [-I, A]], format="csr")
Bsym = sparse.bmat([[Z, I], [-I, A]], format="csr")
S = sparse.bmat([[DIinv, Z], [Z, I]], format="csr")
Sinv = sparse.bmat([[ (D-I), Z], [Z, I]], format="csr")

dim = 5 

s2, v2 = sparse.linalg.eigs(B, k=dim + 1, tol = 1e-8 * 0)
v2 = np.real(v2)
s2 = np.real(s2)
l= np.linalg.norm(v2, axis=0)
s2 = s2 * l
v2 = np.einsum("ij,j->ij", v2, 1/l)
print(s2)
