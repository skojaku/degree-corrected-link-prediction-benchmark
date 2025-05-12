# %%
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import rankdata
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import csgraph
from heart.pagerank import calc_ppr_forward_push_fast
import numba

@numba.jit(nopython=True)
def fast_rankdata(x):
    """Simple implementation of rankdata that works with Numba"""
    temp = x.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(1, len(x) + 1)
    return ranks

@numba.jit(nopython=True)
def fast_rank_scores(scores, mask_indices):
    """Optimized ranking function using Numba with efficient masking"""
    scores = scores.copy()
    scores[mask_indices] = -1
    mask_positive = scores > 0

    if np.sum(mask_positive) > 0:
        positive_scores = scores[mask_positive]
        ranks = fast_rankdata(-positive_scores)
        max_rank = ranks.max()
        scores[mask_positive] = ranks
        scores[~mask_positive] = max_rank + 1
    return scores

@numba.jit(nopython=True)
def get_neighbors_csr(indptr, indices, node):
    """Get neighbors of a node using CSR format arrays"""
    start, end = indptr[node], indptr[node + 1]
    return indices[start:end]

class HeartSampler:
    def __init__(self, testEdgeFraction, alpha=0.15, eps=5e-5, num_neg_samples=1, cn_metric="RA", agg="min"):
        self.alpha = alpha
        self.eps = eps
        self.num_neg_samples = num_neg_samples
        self.cn_metric = cn_metric
        self.agg = np.mean if agg == "mean" else np.min
        self.testEdgeFraction = testEdgeFraction

    def _calc_ppr_scores(self, adj):
        n = adj.shape[0]
        deg = np.array(adj.sum(1)).flatten()
        deg_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
        norm_adj = deg_inv @ adj
        identity = sp.identity(n)
        #ppr = (1 - self.alpha) * sp.linalg.inv(identity - self.alpha * norm_adj)
        ppr = calc_ppr_forward_push_fast(adj, alpha = self.alpha)
        return ppr

    def _calc_cn_scores(self, adj):
        if self.cn_metric == "RA":
            deg = np.array(adj.sum(1)).flatten()
            deg_inv = sp.diags(1.0 / np.maximum(deg, 1e-12))
            weighted_adj = adj @ deg_inv
            return adj @ weighted_adj
        return adj @ adj

    def _get_mask_indices(self, source, target, indptr, indices):
        """Get indices of entries to mask using CSR format"""
        # Get neighbors efficiently using CSR format
        source_neighbors = get_neighbors_csr(indptr, indices, source)
        target_neighbors = get_neighbors_csr(indptr, indices, target)

        # Combine all indices to mask (neighbors + self-loops)
        mask_indices = np.unique(np.concatenate([
            source_neighbors,
            target_neighbors,
            [source],
            [target]
        ]))

        return mask_indices

    def _batch_negative_sampling(self, test_edges, adj, cn_scores, ppr_scores, batch_size=10000):
        """Process negative samples in batches with efficient CSR format usage"""
        n = adj.shape[0]
        all_neg_edges = []

        # Ensure adj is in CSR format and get its arrays
        adj_csr = adj.tocsr()
        indptr = adj_csr.indptr
        indices = adj_csr.indices

        # Convert score matrices to dense for batch processing
        cn_dense = cn_scores.toarray()
        ppr_dense = ppr_scores.toarray()

        batch_size = np.min([batch_size, len(test_edges)])

        for i in range(0, len(test_edges), batch_size):
            batch_edges = test_edges[i:i + batch_size]
            batch_sources = batch_edges[:, 0]
            batch_targets = batch_edges[:, 1]

            # Get scores for batch
            source_cn = cn_dense[batch_sources]
            source_ppr = ppr_dense[batch_sources]
            target_cn = cn_dense[batch_targets]
            target_ppr = ppr_dense[batch_targets]

            # Process each edge in batch
            for j, (src, trg) in enumerate(zip(batch_sources, batch_targets)):
                # Get mask indices using CSR format
                mask_indices = self._get_mask_indices(src, trg, indptr, indices)

                # Rank scores using efficient masking
                src_cn_ranks = fast_rank_scores(source_cn[j], mask_indices)
                src_ppr_ranks = fast_rank_scores(source_ppr[j], mask_indices)
                trg_cn_ranks = fast_rank_scores(target_cn[j], mask_indices)
                trg_ppr_ranks = fast_rank_scores(target_ppr[j], mask_indices)

                # Combine ranks
                src_combined = self.agg(np.column_stack((src_cn_ranks, src_ppr_ranks)), axis=1)
                trg_combined = self.agg(np.column_stack((trg_cn_ranks, trg_ppr_ranks)), axis=1)

                # Select top-k negative samples
                k = self.num_neg_samples
                src_negs = np.argpartition(src_combined, k)[:k]
                trg_negs = np.argpartition(trg_combined, k)[:k]

                # Create negative edges
                src_edges = np.column_stack((np.repeat(src, k), src_negs))
                trg_edges = np.column_stack((trg_negs, np.repeat(trg, k)))
                all_neg_edges.append(np.vstack((src_edges, trg_edges)))
        neg_edges = np.vstack(all_neg_edges)
        return neg_edges[np.random.choice(len(neg_edges), size = self.num_neg_samples *len(neg_edges) // 2, replace=False), :]

    def generate_samples(self, adj, test_edges=None, features=None):
        """Generate positive and negative samples efficiently"""
        if test_edges is None:
            splitter = TrainTestEdgeSplitter(fraction=self.testEdgeFraction)
            splitter.fit(adj)
            test_edges = np.vstack(splitter.test_edges_).T

        print("Calculating PPR scores...")
        ppr_scores = self._calc_ppr_scores(adj)

        print(f"Calculating {self.cn_metric} scores...")
        cn_scores = self._calc_cn_scores(adj)

        print("Generating negative samples...")
        neg_edges = self._batch_negative_sampling(test_edges, adj, cn_scores, ppr_scores)

        # Combine positive and negative edges
        src = np.concatenate([test_edges[:, 0], neg_edges[:, 0]])
        trg = np.concatenate([test_edges[:, 1], neg_edges[:, 1]])
        y = np.concatenate([np.ones(len(test_edges)), np.zeros(len(neg_edges))])

        return src, trg, y

    # Forward push method for very large graphs
    def _calc_ppr_forward_push(self, adj, alpha=0.15, epsilon=1e-6):
        """Calculate PPR scores using forward push algorithm"""
        n = adj.shape[0]
        adj_csr = adj.tocsr()

        # Normalize adjacency matrix
        deg = np.array(adj_csr.sum(1)).flatten()
        deg_inv = 1.0 / np.maximum(deg, 1e-12)

        ppr_data = []
        ppr_rows = []
        ppr_cols = []

        for source in tqdm(range(n), desc="Computing PPR"):
            # Initialize residual and ppr vectors
            residual = np.zeros(n)
            residual[source] = 1.0
            ppr = np.zeros(n)

            # Process nodes with residual > threshold
            active = {source}
            while active:
                node = active.pop()
                push_value = residual[node]
                residual[node] = 0

                # Add to PPR
                ppr[node] += (1 - alpha) * push_value

                # Distribute remaining residual
                if adj_csr.indptr[node] != adj_csr.indptr[node + 1]:
                    neighbors = adj_csr.indices[adj_csr.indptr[node]:adj_csr.indptr[node + 1]]
                    neighbor_vals = alpha * push_value * deg_inv[node]
                    residual[neighbors] += neighbor_vals

                    # Add high residual nodes to active set
                    new_active = neighbors[residual[neighbors] > epsilon * deg[neighbors]]
                    active.update(new_active)

            # Store significant entries
            significant_indices = np.where(ppr > epsilon)[0]
            ppr_data.extend(ppr[significant_indices])
            ppr_rows.extend([source] * len(significant_indices))
            ppr_cols.extend(significant_indices)

        return sp.csr_matrix((ppr_data, (ppr_rows, ppr_cols)), shape=(n, n))

class TrainTestEdgeSplitter:
    def __init__(self, fraction=0.25):
        """
        Initialize the TrainTestEdgeSplitter with a specified fraction of edges to be used as test edges.

        Parameters
        ----------
        fraction : float, optional
            The fraction of edges to be removed from the training graph and used as test edges. Default is 0.25.

        Examples
        --------
        >>> from scipy.sparse import csr_matrix
        >>> A = csr_matrix([
        ...     [0, 1, 0, 0],
        ...     [1, 0, 1, 0],
        ...     [0, 1, 0, 1],
        ...     [0, 0, 1, 0]
        ... ])
        >>> splitter = TrainTestEdgeSplitter(fraction=0.2)
        >>> splitter.fit(A)
        >>> test_edges = splitter.test_edges_
        >>> train_edges = splitter.train_edges_
        """
        self.fraction = fraction

    def __init__(self, fraction=0.25):
        """
        Initialize the TrainTestEdgeSplitter with a fraction of edges to be used as test edges.

        Parameters
        ----------
        fraction : float
            The fraction of edges to be removed from the graph and used as test edges.

        Attributes
        ----------
        fraction : float
            Stores the fraction of edges that will be used as test edges.
        """

        self.fraction = fraction

    def fit(self, A):
        """
        Fit the splitter to the adjacency matrix and prepare for edge splitting.

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
            The adjacency matrix of the graph.

        Returns
        -------
        None
        """

        r, c, _ = sparse.find(A)
        edges = np.unique(pairing(r, c))

        MST = csgraph.minimum_spanning_tree(A + A.T)
        r, c, _ = sparse.find(MST)
        mst_edges = np.unique(pairing(r, c))
        remained_edge_set = np.array(
            list(set(list(edges)).difference(set(list(mst_edges))))
        )
        n_edge_removal = int(len(edges) * self.fraction)
        if len(remained_edge_set) < n_edge_removal:
            raise Exception(
                "Cannot remove edges by keeping the connectedness. Decrease the `fraction` parameter"
            )

        test_edge_set = np.random.choice(
            remained_edge_set, n_edge_removal, replace=False
        )

        train_edge_set = np.array(
            list(set(list(edges)).difference(set(list(test_edge_set))))
        )

        self.test_edges_ = depairing(test_edge_set)
        self.train_edges_ = depairing(train_edge_set)
        self.n = A.shape[0]

    def transform(self):
        """
        Transform the graph by splitting it into training and testing edges.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing two elements:
            - train_edges_: Array of training edges.
            - test_edges_: Array of testing edges.
        """

        return self.train_edges_, self.test_edges_


def pairing(r, c):
    """
    Pair two arrays of row and column indices into a single array of complex numbers.

    Parameters
    ----------
    r : np.ndarray
        Array of row indices.
    c : np.ndarray
        Array of column indices.

    Returns
    -------
    np.ndarray
        Array of complex numbers where each complex number represents a pair (r, c).
    """
    return np.minimum(r, c) + 1j * np.maximum(r, c)


def depairing(v):
    """
    Decompose a complex number array into its real and imaginary parts as integers.

    Parameters
    ----------
    v : np.ndarray
        Array of complex numbers.

    Returns
    -------
    tuple
        A tuple containing two np.ndarray:
        - The first array contains the real parts of the complex numbers.
        - The second array contains the imaginary parts of the complex numbers.
    """

    return np.real(v).astype(int), np.imag(v).astype(int)


def to_adjacency_matrix(net):
    """
    Convert a given network into an adjacency matrix.

    Parameters
    ----------
    net : various types
        The network data which can be a scipy sparse matrix, networkx graph, or numpy ndarray.

    Returns
    -------
    tuple
        A tuple containing the adjacency matrix and a string indicating the type of the input network.
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net, "scipy.sparse"
        return sparse.csr_matrix(net), "scipy.sparse"
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net), "networkx"
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net), "numpy.ndarray"


def transform_network_data_type(net, to_type, from_type="scipy.sparse"):
    """
    Convert the network data type.

    Parameters
    ----------
    net : various types
        The network data which might be in different formats like scipy sparse matrix, networkx graph, or numpy array.
    to_type : str
        The target type to which the network data should be converted. Options are 'scipy.sparse', 'networkx', 'numpy.ndarray'.
    from_type : str, optional
        The original type of the network data. Default is 'scipy.sparse'.

    Returns
    -------
    converted_net : various types
        The network data converted to the specified type.
    """

    if from_type != "scipy.sparse":
        _net, _ = to_adjacency_matrix(net)
    else:
        _net = net

    if to_type == "scipy.sparse":
        return _net
    elif to_type == "networkx":
        return nx.from_scipy_sparse_array(_net)
    elif to_type == "numpy.ndarray":
        return _net.toarray()

#import networkx as nx
#import heart
#import numpy as np
#from scipy.sparse import csr_matrix
#
#G = nx.karate_club_graph()
#A = nx.adjacency_matrix(G)
#labels = np.unique([d[1]["club"] for d in G.nodes(data=True)], return_inverse=True)[1]
#
#sampler = HeartSampler(testEdgeFraction=0.1)
#sampler.generate_samples(sparse.csr_matrix(A))
# %%

# %%