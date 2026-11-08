"""Clustering on parameters and QoIs, the cluster tree, and cluster assignment.

The initial clusters come from k-means on the scaled feature vectors ``[params | QoI]``, and are then refined by splitting clusters in two.

At solve time a feature vector is assigned a cluster by descending the cluster tree, which reproduces the clusters the bases were fitted on. Taking the nearest leaf centroid is a different map and can assign a point to a basis fitted without it.
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

QOI_FLOOR = 1e-30            # clip before log10; a zero abundance breaks k-means


def block_starts(tracer, step):
    """Positions starting a new ``(tracer, step)`` block (one trajectory on one interval) in a row-ordered sequence.

    Keyed on both columns: ``step`` alone cannot separate two trajectories when the grid has one interval per trajectory, which silently merges them into one block.
    """
    tracer, step = np.asarray(tracer), np.asarray(step)
    out = np.empty(step.size, dtype=bool)
    out[0] = True
    out[1:] = (step[1:] != step[:-1]) | (tracer[1:] != tracer[:-1])
    return out


class Cluster:
    """k-means clusters on parameters and QoIs, with the recorded cluster tree."""

    def __init__(self, param_names=()):
        self.param_names = list(param_names)
        self._log_cols = None        # param columns log10'd, original indexing
        self._feature_cols = None    # param columns kept as features
        self.qoi_indices = None      # state columns joining the params
        self.qoi_log = True          # log10 the QoI features before scaling
        self.feature_mean = None     # z-scoring of the [params | QoI] features
        self.feature_scaling = None

        self.labels = None           # one cluster id per feature vector
        self.n_clusters = None
        self.centroids = None        # (k, n_features); NaN row = empty cluster
        self._features = None        # the scaled matrix that was clustered

        # Cluster tree, as parallel arrays over nodes. ``centroids`` holds leaves only -- an internal node's row is overwritten by its own children -- so each node's split-time centroid is recorded here.
        self.tree_left = None        # child node index, or -1 for a leaf
        self.tree_right = None
        self.tree_cluster = None     # cluster id at a leaf, -1 when internal
        self.tree_centroid = None
        self._roots = None           # tree nodes nothing points to

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, fm, cluster_indices, k, log_cols, drop_cols, qoi_log=True,
            seed=0, offset=None):
        """Fit the initial clusters on the parameters plus the state columns in ``cluster_indices``.

        ``cluster_indices`` need not be the scored QoIs: Lorenz-96 clusters on all components while scoring four of them.

        Fitted on one feature vector per trajectory and interval, from the state at the start of the interval, which is the state :meth:`predict` receives at solve time; a clustering built from a training-only ``fm`` therefore uses training data only.

        ``qoi_log`` log10s the state features, putting positive abundances spanning decades on a comparable footing with the parameters. A signed state must set it False.

        ``offset`` is the network's :meth:`offset`: when it returns an array for the parameter rows, the state features are formed from ``x = x0 + state``, the variable :meth:`predict` is then called with.
        """
        init_idx = np.flatnonzero(block_starts(fm.sample_tracer, fm.sample_step))

        cluster_indices = np.asarray(cluster_indices, dtype=int)
        cols = np.concatenate([
            np.arange(fm.param_slice.start, fm.param_slice.stop),
            fm.species_slice.start + cluster_indices,
        ])
        block = np.asarray(fm.memmap[np.ix_(init_idx, cols)], dtype=np.float64)
        raw_params, raw_qoi = block[:, :fm.n_params], block[:, fm.n_params:]
        x0 = None if offset is None else offset(raw_params)
        if x0 is not None:
            raw_qoi = raw_qoi + x0[:, cluster_indices]

        drop = set(self._resolve(drop_cols))
        self._feature_cols = [i for i in range(fm.n_params) if i not in drop]
        self._log_cols = [c for c in self._resolve(log_cols) if c not in drop]

        p = raw_params.copy()
        p[:, self._log_cols] = np.log10(p[:, self._log_cols])
        self.qoi_indices = cluster_indices
        self.qoi_log = bool(qoi_log)
        q = np.log10(np.clip(raw_qoi, QOI_FLOOR, None)) if self.qoi_log else raw_qoi

        scaler = StandardScaler().fit(np.column_stack([p[:, self._feature_cols], q]))
        self.feature_mean, self.feature_scaling = scaler.mean_, scaler.scale_
        joint = scaler.transform(np.column_stack([p[:, self._feature_cols], q]))
        print(f"Joint clustering ({init_idx.size:,} steps, "
              f"{len(self._feature_cols)} params + {cluster_indices.size} "
              f"state components), k-means k={k} ...")
        self.labels = np.asarray(
            KMeans(n_clusters=int(k), n_init="auto",
                   random_state=seed).fit_predict(joint))
        self.n_clusters = int(k)
        self._features = joint
        self.centroids = self._means(joint, self.labels, int(k))

        # The initial clusters are the roots; refinement adds subtrees under them.
        self.tree_left = np.full(k, -1, dtype=np.int64)
        self.tree_right = np.full(k, -1, dtype=np.int64)
        self.tree_cluster = np.arange(k, dtype=np.int64)
        self.tree_centroid = self.centroids.copy()
        # A split only appends children, so these nodes remain the roots.
        self._roots = np.arange(k, dtype=np.int64)
        return self.labels

    @staticmethod
    def _means(features, labels, k):
        """Per-cluster mean in scaled space; NaN row for an empty cluster."""
        out = np.full((k, features.shape[1]), np.nan, dtype=np.float64)
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                out[c] = features[mask].mean(axis=0)
        return out

    def clone(self):
        """Copy that shares the large arrays and has its own partition.

        ``labels``, ``centroids`` and the tree are the only state a split modifies, so one base fit can be the starting point for every split tolerance.
        """
        new = copy.copy(self)
        new.labels = self.labels.copy()
        new.centroids = self.centroids.copy()
        for a in ("tree_left", "tree_right", "tree_cluster", "tree_centroid"):
            setattr(new, a, np.asarray(getattr(self, a)).copy())
        return new

    def split_cluster(self, c, rng, min_child_size):
        """Split cluster ``c`` in two with k-means (k=2) on its feature vectors.

        One child keeps the id ``c`` and the other gets a new id, which is returned. Returns ``None`` without changing the clusters if either child would be smaller than ``min_child_size``, the minimum cluster size.
        """
        rows = np.flatnonzero(self.labels == c)
        if rows.size < 2 * min_child_size:
            return None
        sub = self._features[rows]
        if np.unique(sub, axis=0).shape[0] < 2:
            return None

        seed = int(rng.integers(0, 2 ** 31 - 1))
        child = np.asarray(
            KMeans(n_clusters=2, n_init="auto", random_state=seed).fit_predict(sub))
        if min(np.count_nonzero(child == 0), np.count_nonzero(child == 1)) \
                < min_child_size:
            return None

        new_id = int(self.n_clusters)
        self.labels[rows[child == 1]] = new_id
        self.n_clusters = new_id + 1

        grown = np.full((self.n_clusters, self._features.shape[1]), np.nan)
        grown[:self.centroids.shape[0]] = self.centroids
        self.centroids = grown
        for cid in (c, new_id):
            self.centroids[cid] = self._features[self.labels == cid].mean(axis=0)
        self._record_split(c, new_id)
        return new_id

    def _record_split(self, c, new_id):
        """Turn leaf ``c`` into an internal node with children ``c``/``new_id``."""
        hit = np.flatnonzero(self.tree_cluster == int(c))
        node = int(hit[0])
        first = self.tree_left.size
        self.tree_left = np.append(self.tree_left, [-1, -1])
        self.tree_right = np.append(self.tree_right, [-1, -1])
        self.tree_cluster = np.append(self.tree_cluster, [int(c), int(new_id)])
        # Captured now: a child that is itself split later has its centroids row overwritten by its own children.
        self.tree_centroid = np.vstack(
            [self.tree_centroid, self.centroids[[c, new_id]]])
        self.tree_left[node] = first
        self.tree_right[node] = first + 1
        self.tree_cluster[node] = -1

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def predict(self, p, x):
        """Cluster for parameter row ``p`` and current state ``x``.

        With a single cluster, everything is assigned to it without forming the feature vector, so a k=1 fit stays usable against a parameter row of a different width than it was fitted on.
        """
        if self.n_clusters == 1:
            return int(self.tree_cluster[0])
        return self._route_tree(self.routing_feature(p, x))

    def routing_feature(self, p, x):
        """The scaled feature vector ``[params | QoI]`` used to descend the cluster tree.

        Z-scored inline rather than through sklearn: this runs once per interval per trajectory on a single row, and ``transform`` spends most of its time validating.
        """
        p = np.asarray(p, dtype=np.float64).copy()
        p[self._log_cols] = np.log10(p[self._log_cols])
        q = np.asarray(x, dtype=np.float64)[self.qoi_indices]
        if self.qoi_log:
            q = np.log10(np.clip(q, QOI_FLOOR, None))
        return (np.concatenate([p[self._feature_cols], q]) - self.feature_mean) / self.feature_scaling

    def _route_tree(self, feat):
        """Nearest root centroid, then descend to a leaf."""
        d2 = ((self.tree_centroid[self._roots] - feat) ** 2).sum(axis=1)
        node = int(self._roots[int(np.argmin(np.where(np.isnan(d2), np.inf, d2)))])
        while self.tree_left[node] >= 0:
            a, b = int(self.tree_left[node]), int(self.tree_right[node])
            da = np.sum((self.tree_centroid[a] - feat) ** 2)
            db = np.sum((self.tree_centroid[b] - feat) ** 2)
            node = a if (np.nan_to_num(da, nan=np.inf)
                         <= np.nan_to_num(db, nan=np.inf)) else b
        return int(self.tree_cluster[node])

    def labels_for_rows(self, fm):
        """Per-row cluster labels: each row takes the label of its trajectory and interval."""
        starts = block_starts(fm.sample_tracer, fm.sample_step)
        return np.asarray(self.labels)[np.cumsum(starts) - 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state(self):
        """The arrays cluster assignment needs: the feature scaling, the cluster tree and the centroids.

        Neither the clustered feature vectors nor the training labels are included: they are needed only to fit and refine, so a reloaded tree assigns clusters but cannot be refined further.
        """
        return dict(
            feature_mean=self.feature_mean,
            feature_scaling=self.feature_scaling,
            log_cols=np.asarray(self._log_cols),
            feature_cols=np.asarray(self._feature_cols),
            qoi_indices=self.qoi_indices,
            qoi_log=np.array(self.qoi_log),
            n_clusters=np.array(self.n_clusters),
            centroids=self.centroids,
            tree_left=self.tree_left,
            tree_right=self.tree_right,
            tree_cluster=self.tree_cluster,
            tree_centroid=self.tree_centroid,
        )

    @classmethod
    def from_state(cls, d):
        """The cluster tree described by the arrays of :meth:`state`, ready to assign clusters."""
        self = cls()
        self.feature_mean = np.asarray(d["feature_mean"])
        self.feature_scaling = np.asarray(d["feature_scaling"])
        self._log_cols = np.asarray(d["log_cols"]).tolist()
        self._feature_cols = np.asarray(d["feature_cols"]).tolist()
        self.qoi_indices = np.asarray(d["qoi_indices"]).astype(int)
        self.qoi_log = bool(d["qoi_log"])
        self.n_clusters = int(d["n_clusters"])
        self.centroids = np.asarray(d["centroids"])
        self.tree_left = np.asarray(d["tree_left"]).astype(np.int64)
        self.tree_right = np.asarray(d["tree_right"]).astype(np.int64)
        self.tree_cluster = np.asarray(d["tree_cluster"]).astype(np.int64)
        self.tree_centroid = np.asarray(d["tree_centroid"])
        # Roots are the nodes nothing points to; resolve once, not per interval.
        nodes = np.arange(self.tree_left.size)
        children = np.concatenate([self.tree_left[self.tree_left >= 0],
                                   self.tree_right[self.tree_right >= 0]])
        self._roots = nodes[~np.isin(nodes, children)]
        return self

    # ------------------------------------------------------------------

    def _resolve(self, cols):
        """Column names or indices -> integer indices."""
        out = []
        for c in cols or ():
            if isinstance(c, str):
                if c not in self.param_names:
                    raise ValueError(f"unknown parameter {c!r}; valid names: "
                                     f"{self.param_names}")
                out.append(self.param_names.index(c))
            else:
                out.append(int(c))
        return out
