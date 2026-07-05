import unittest

import numpy as np

from funasr.models.campplus.cluster_backend import ClusterBackend


def make_blobs(centers, points_per_cluster, noise_scale=0.01, seed=0):
    """Synthetic per-cluster embeddings tightly grouped around each center."""
    rng = np.random.default_rng(seed)
    labels = []
    vectors = []
    for i, center in enumerate(centers):
        pts = center + rng.normal(scale=noise_scale, size=(points_per_cluster, len(center)))
        vectors.append(pts)
        labels.extend([i] * points_per_cluster)
    return np.concatenate(vectors, axis=0), np.array(labels)


class TestMergeToMaxSpks(unittest.TestCase):
    def test_merges_closest_pair_first(self):
        # Cluster 2 and 3 are near-identical directions (cosine ~1); 0 and 1 are far apart.
        # Capping to 3 speakers must merge 2 and 3 together, leaving 0 and 1 untouched.
        centers = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.02],
        ]
        embs, labels = make_blobs(centers, points_per_cluster=5)

        cb = ClusterBackend()
        merged = cb.merge_to_max_spks(labels.copy(), embs, max_num_spks=3)

        self.assertEqual(len(set(merged.tolist())), 3)
        # points originally in cluster 2 and cluster 3 must now share one label
        self.assertEqual(merged[labels == 2][0], merged[labels == 3][0])
        # clusters 0 and 1 (mutually farthest) must stay distinct from each other
        self.assertNotEqual(merged[labels == 0][0], merged[labels == 1][0])

    def test_noop_when_already_within_cap(self):
        centers = [[1.0, 0.0], [0.0, 1.0]]
        embs, labels = make_blobs(centers, points_per_cluster=5)

        cb = ClusterBackend()
        merged = cb.merge_to_max_spks(labels.copy(), embs, max_num_spks=5)

        np.testing.assert_array_equal(merged, labels)


class TestClusterBackendForwardRespectsCapOnLargeInput(unittest.TestCase):
    def test_umap_path_is_capped_by_max_num_spks(self):
        # >=2048 rows routes forward() to umap_hdbscan_cluster, which has no speaker-count
        # concept at all. Stub it out (real UMAP/HDBSCAN is slow + irrelevant to this bound
        # check) so the test exercises only the cap-enforcement wiring in forward().
        centers = [[float(i == j) for j in range(8)] for i in range(8)]
        embs, raw_labels = make_blobs(centers, points_per_cluster=300)  # 2400 rows, 8 clusters

        cb = ClusterBackend()
        cb.spectral_cluster.max_num_spks = 5
        cb.umap_hdbscan_cluster = lambda X: raw_labels.copy()

        result = cb.forward(embs)

        self.assertLessEqual(len(set(result.tolist())), 5)


if __name__ == "__main__":
    unittest.main()
