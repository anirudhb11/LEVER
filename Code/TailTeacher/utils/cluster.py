import numpy as np
from xclib.utils.clustering import cluster_balance, b_kmeans_dense
from scipy.sparse import csr_matrix

def cluster_items(X, depth, n_threads):
    n_clusters = 2**(depth-1)
    clusters, _ = cluster_balance(
        X=X.copy(), 
        clusters=[np.arange(len(X), dtype=np.int32)],
        num_clusters=n_clusters,
        splitter=b_kmeans_dense,
        num_threads=n_threads,
        verbose=True)
    clustering_mat = csr_matrix((np.ones(sum([len(c) for c in clusters])), 
                                     np.concatenate(clusters),
                                     np.cumsum([0, *[len(c) for c in clusters]])),
                                 shape=(len(clusters), X.shape[0]))
    return clustering_mat

