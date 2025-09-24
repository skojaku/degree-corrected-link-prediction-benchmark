import numpy as np
from scipy import sparse
from tqdm import tqdm
import numba
from numba import jit, float64, int64
from numba.experimental import jitclass

@jit(nopython=True)
def _push_from_node(node, residual, ppr, neighbors_indptr, neighbors_indices,
                    deg_inv, alpha, epsilon, deg):
    """Optimized single node push operation"""
    push_value = residual[node]
    residual[node] = 0

    # Add to PPR
    ppr[node] += (1 - alpha) * push_value

    # Get node's neighbors using CSR format
    start = neighbors_indptr[node]
    end = neighbors_indptr[node + 1]

    # Only push if node has neighbors
    if start != end:
        # Push to neighbors
        neighbor_update = alpha * push_value * deg_inv[node]
        neighbors = neighbors_indices[start:end]

        # Update residuals and collect new active nodes
        new_active = []
        for neighbor in neighbors:
            old_residual = residual[neighbor]
            residual[neighbor] += neighbor_update

            # Check if node becomes active
            if old_residual <= epsilon * deg[neighbor] and residual[neighbor] > epsilon * deg[neighbor]:
                new_active.append(neighbor)

    return new_active

@jit(nopython=True)
def _forward_push_single_source(source, n, neighbors_indptr, neighbors_indices,
                              deg_inv, deg, alpha, epsilon):
    """Compute PPR scores for a single source node"""
    residual = np.zeros(n)
    ppr = np.zeros(n)

    # Initialize residual at source
    residual[source] = 1.0

    # Active nodes queue - implement as array for Numba compatibility
    active = np.zeros(n, dtype=np.int64)
    active[0] = source
    active_size = 1

    while active_size > 0:
        # Pop last active node
        active_size -= 1
        node = active[active_size]

        # Process node
        new_active = _push_from_node(node, residual, ppr, neighbors_indptr,
                                   neighbors_indices, deg_inv, alpha, epsilon, deg)

        # Add new active nodes
        for new_node in new_active:
            if active_size < n:  # Prevent overflow
                active[active_size] = new_node
                active_size += 1

    return ppr

def calc_ppr_forward_push_fast(adj, alpha=0.15, epsilon=1e-6, batch_size=1000):
    """
    Compute PPR using numba-accelerated forward push method.

    Parameters:
    -----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix
    alpha : float
        Teleportation probability
    epsilon : float
        Tolerance threshold
    batch_size : int
        Number of sources to process in parallel
    """
    n = adj.shape[0]
    adj_csr = adj.tocsr()

    # Precompute values needed for all sources
    deg = np.array(adj_csr.sum(1)).flatten()
    deg_inv = 1.0 / np.maximum(deg, 1e-12)

    # Get CSR format arrays
    neighbors_indptr = adj_csr.indptr
    neighbors_indices = adj_csr.indices

    # Storage for sparse PPR matrix
    ppr_data = []
    ppr_rows = []
    ppr_cols = []

    # Process sources in batches
    for batch_start in tqdm(range(0, n, batch_size), desc="Computing PPR"):
        batch_end = min(batch_start + batch_size, n)
        batch_sources = range(batch_start, batch_end)

        # Compute PPR for batch
        for source in batch_sources:
            ppr = _forward_push_single_source(
                source, n, neighbors_indptr, neighbors_indices,
                deg_inv, deg, alpha, epsilon
            )

            # Store significant entries
            significant_indices = np.where(ppr > epsilon)[0]
            ppr_data.extend(ppr[significant_indices])
            ppr_rows.extend([source] * len(significant_indices))
            ppr_cols.extend(significant_indices)

    # Create sparse matrix
    return sparse.csr_matrix(
        (ppr_data, (ppr_rows, ppr_cols)),
        shape=(n, n)
    )

# Additional utility for parallel processing multiple sources
@jit(nopython=True, parallel=True)
def _forward_push_batch(sources, n, neighbors_indptr, neighbors_indices,
                       deg_inv, deg, alpha, epsilon):
    """Process multiple source nodes in parallel"""
    batch_size = len(sources)
    results = np.zeros((batch_size, n))

    for i in numba.prange(batch_size):
        results[i] = _forward_push_single_source(
            sources[i], n, neighbors_indptr, neighbors_indices,
            deg_inv, deg, alpha, epsilon
        )

    return results