import enum
from re import VERBOSE
import numpy as np
import numba as nb
import time
from pynndescent import NNDescent
from sklearn.neighbors import KDTree, BallTree,NearestNeighbors
from Dataprocessing import normalize_by_minmax
import joblib

def ts():
    return time.ctime(time.time())


@nb.njit(parallel=True)
def fast_knn_indices(X, n_neighbors):
    """A fast computation of knn indices.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor indices of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    knn_indices = np.empty((X.shape[0], n_neighbors), dtype=np.int32)
    for row in nb.prange(X.shape[0]):
        # v = np.argsort(X[row])  # Need to call argsort this way for numba
        v = X[row].argsort(kind="quicksort")
        v = v[:n_neighbors]
        knn_indices[row] = v
    return knn_indices

def nearest_neighbors(
        X,
        n_neighbors,
        metric,
        metric_kwds,
        angular,
        random_state,
        low_memory=True,
        use_pynndescent=True,
        n_jobs=-1,
        verbose=False,
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metri
    /c: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    low_memory: bool (optional, default True)
        Whether to pursue lower memory NNdescent.

    verbose: bool (optional, default False)
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    rp_forest: list of trees
        The random projection forest used for searching (if used, None otherwise)
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    search_term = None
    if metric == "precomputed":
        search_term = metric
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = fast_knn_indices(X, n_neighbors)
        # knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        # Prune any nearest neighbours that are infinite distance apart.
        disconnected_index = knn_dists == np.inf
        knn_indices[disconnected_index] = -1

        knn_search_index = None
    elif X[0].shape[0] < 100000:
        search_term = 'KDTree'
        # knn_search_index = KDTree(X, metric=metric)
        # query_res = knn_search_index.query(X, k=n_neighbors + 1)
        knn_search_index=NearestNeighbors(algorithm='kd_tree').fit(X)
        query_res = knn_search_index.kneighbors(X,n_neighbors=n_neighbors + 1)
        knn_indices = query_res[1][:, 1:]
        knn_dists = query_res[0][:, 1:]
    else:
        search_term = 'Annoy + NND'
        # TODO: Hacked values for now
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        knn_search_index = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_kwds=metric_kwds,
            random_state=random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        knn_indices, knn_dists = knn_search_index.neighbor_graph

    if verbose:
        print(ts(), "Finished Nearest Neighbor Search By", search_term)
    return knn_indices, knn_dists, knn_search_index


class FastDEC:
    """
    Parameters
    ----------

    k: The number of neighbors (i.e. the k in k-NN density)

    n_clusters: number of clustering result

    metric:

    Attributes
    ----------

    cluster_map: a map from the cluster (zero-based indexed) to the list of points
        in that cluster

    """

    def __init__(self, k, n_clusters = 0,  ann="kdtree", metric="euclidean", denstiy_estimator='gaussion',
                knn_info=None):
        """
            knn_info: knn_info dict, stored knn path, default None, None is to rebuild knn
            space index and not None is directly using knn_info, knn_info dict definition:\n
            knn_info={
                "search_index":knn_path+data_name+".joblib",
                "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
                "max_neigh":max_neigh
            }, the purpose of knn_info is to speedup algorithm.
        """
        self.k = k
        self.n_clusters = n_clusters
        self.density_estimator = denstiy_estimator
        self.ann = ann
        self.metric = metric
        self.knn_info=knn_info  # lhz add knn_info
    def query_kNN(self, X, k, metric = 'euclidean'):
        if self.knn_info is None or self.knn_info["max_neigh"]<k+1:
            verbose = True
            random_state = 2022
            _knn_indices, _knn_dists, searh_index = nearest_neighbors(X,
                k,
                metric,
                {},
                True,
                random_state,
                verbose=verbose,
                )
            self.searh_index = searh_index
        else:
            self.searh_index=joblib.load(self.knn_info["search_index"])
            knn_dists_indices = np.load(self.knn_info["dists_indices"])
            _knn_indices, _knn_dists=knn_dists_indices["knn_indices"],knn_dists_indices["knn_dists"]
        return _knn_indices[:,1:k+1], _knn_dists[:,1:k+1]
    def density_estimation(self,_knn_dists):
        # density = np.sum(np.exp(-_knn_dists / 2 / mean_vale), axis=1)
        # density = - _knn_dists[:, k - 1]
        if self.density_estimator == 'gaussion':
            mean_vale = np.mean(_knn_dists)
            density = np.sum(np.exp(-_knn_dists / 2 / mean_vale), axis=1)
        if self.density_estimator == 'kth':
            density = 1/_knn_dists[:, self.k - 1]
        if self.density_estimator == 'flat':
            mean_vale = np.mean(_knn_dists)
            density = np.full(self.n, 0 , np.int32)
            for i in range(self.n):
                idx=  np.where(_knn_dists[i] < mean_vale)[0]
                density[i] = len(idx)
        return density


    def DC_detection(self, density, _knn_indices, _knn_dists):
        cdh_ids = np.full(self.n, -1, dtype=np.int32)
        k_int = np.full(self.n, self.n, dtype = np.int32)

        for i in range(self.n):
            for q,j in enumerate(_knn_indices[i]):
                if density[i] < density[j]:
                    cdh_ids[i] = j
                    k_int[i] = q + 1
                    break
        core_idx = np.where(cdh_ids == -1)[0]
        n_cores = len(core_idx)
        self.k_int = k_int / self.n
        return cdh_ids, core_idx, n_cores

    # def DC_inter_dominance_estimation(self, X,density, core_idx):
    #     n_cores = len(core_idx)
    #     # # print(n_cores)
    #     core_cdh = np.full(n_cores, -1 , np.int32)
    #     weight = np.full(n_cores, -1, np.float32)
    #     sorted_idx = np.argsort(-density)
    #     for q,p in enumerate(core_idx):
    #         for j,i in enumerate(sorted_idx):
    #             if i == p:
    #                 if j == 0:
    #                     break
    #                 dist = cdist([X[p]], X[sorted_idx[0:j]])[0]
    #                 min_idx = np.argmin(dist)
    #                 core_cdh[q] = sorted_idx[min_idx]
    #                 weight[q] = dist[min_idx]
    #     for i in range(n_cores):
    #         if weight[i] == -1:
    #             weight[i] = np.max(weight)
    #     self.dist_ndh[core_idx] = weight
    #     weight = normalize_by_minmax(weight)
    #     core_density = density[core_idx]
    #     core_density = normalize_by_minmax(core_density)
    #     SD = core_density * weight
    #     self.weight = weight

    #     topK_idx = core_idx[np.argsort(-SD)[0:self.n_clusters]]
    #     self.topK_idx_core = np.argsort(-SD)[0:self.n_clusters]
    #     self.SD = SD
    #     return cdh_ids, topK_idx

    def DC_inter_dominance_estimation(self, X, density, core_idx, cdh_ids):
        self.density = density
        n_cores = len(core_idx)
        g = np.full(n_cores, -1, np.float32)
        query_res = self.searh_index.kneighbors(X[core_idx,:],n_neighbors=self.n)
        # query_res = self.searh_index.query(X[core_idx,:], k=self.n)
        _knn_indices = query_res[1][:, 1:]
        _knn_dists = query_res[0][:, 1:]
        k_th = np.full(n_cores, -1, np.int32)
        for q, i in enumerate(core_idx):
            for p, j in enumerate(_knn_indices[q]):
                if density[i] < density[j]:
                    g[q] = _knn_dists[q][p]
                    cdh_ids[i] = j
                    k_th[q] = p+1
                    break
        for i in range(n_cores):
            if g[i] == -1:
                g[i] = np.max(g)
            if k_th[i] == -1:
                k_th[i] = np.max(k_th)
        g = normalize_by_minmax(g)
        k_th = normalize_by_minmax(k_th)
        core_density = density[core_idx]
        core_density = normalize_by_minmax(core_density)
        SD = core_density * g * k_th
        self.k_th = k_th
        self.g = g
        self.core_density = core_density
        topK_idx = core_idx[np.argsort(-SD)[0:self.n_clusters]]
        self.topK_idx_core = np.argsort(-SD)[0:self.n_clusters]
        self.SD = SD
        self.topK_idx = topK_idx
        return cdh_ids, topK_idx

    def final_cluster(self, cdh_ids, core_idx, density):
        label = np.full(self.n, -1, np.int32)
        sorted_density = np.argsort(-density)
        count = 0
        for i in core_idx:
            label[i] = count
            count += 1
        for i in sorted_density:
            if label[i] == -1:
                label[i] = label[cdh_ids[i]]
        return label


    def fit(self, data):
        self.n = data.shape[0]
        _knn_indices, _knn_dists = self.query_kNN(data, self.k, self.metric)
        density = self.density_estimation(_knn_dists)
        cdh_ids, core_idx, n_cores = self.DC_detection(density, _knn_indices, _knn_dists)
        self.cdh_ids = cdh_ids.copy()
        self.true_cores = n_cores
        if self.n_clusters == 0:
            self.density = density
            self.core_idx = core_idx
            self.n_cores = n_cores
            label = self.final_cluster(cdh_ids, core_idx, density)
            return label
        else:
            k = self.k
            while self.n_clusters > n_cores:
                k = int(k/2)
                _knn_indices = _knn_indices[:,0:k]
                _knn_dists = _knn_dists[:,0:k]
                density = self.density_estimation(_knn_dists)
                cdh_ids, core_idx, n_cores = self.DC_detection(density, _knn_indices, _knn_dists)
            self.core_idx = core_idx
            self.cdh_ids = cdh_ids.copy()
            self.n_cores = n_cores
        if n_cores == self.n_clusters:
            topK_idx = core_idx
        else:
            cdh_ids,topK_idx = self.DC_inter_dominance_estimation(data, density, core_idx, cdh_ids)
        label = self.final_cluster(cdh_ids, topK_idx, density)
        return label #, density # lhz: 加一个返回值 density





