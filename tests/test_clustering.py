import unittest

import numpy as np

from kmeans import kmeans
from main import format_matrix, run
from symnmf import (
    assign_clusters,
    degree_matrix,
    normalized_similarity_matrix,
    similarity_matrix,
    symnmf_factorization,
)


class SimilarityTests(unittest.TestCase):
    def test_similarity_is_symmetric_with_zero_diagonal(self):
        points = np.array([[0.0], [1.0], [3.0]])
        actual = similarity_matrix(points)
        squared_distances = np.array(
            [[0.0, 1.0, 9.0], [1.0, 0.0, 4.0], [9.0, 4.0, 0.0]]
        )
        expected = np.exp(-squared_distances / 2.0)
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual, actual.T)

    def test_degree_and_normalization_for_isolated_rows(self):
        similarities = np.array(
            [[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        np.testing.assert_allclose(
            np.diag(degree_matrix(similarities)), [2.0, 2.0, 0.0]
        )
        np.testing.assert_allclose(
            normalized_similarity_matrix(similarities),
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )

    def test_invalid_data_is_rejected(self):
        with self.assertRaises(ValueError):
            similarity_matrix([[1.0], [np.nan]])


class ClusteringTests(unittest.TestCase):
    def test_factorization_is_nonnegative_reproducible_and_assignable(self):
        similarities = normalized_similarity_matrix(
            similarity_matrix([[0.0], [0.1], [5.0], [5.1]])
        )
        first = symnmf_factorization(similarities, 2, seed=4)
        second = symnmf_factorization(similarities, 2, seed=4)
        self.assertTrue((first >= 0.0).all())
        np.testing.assert_array_equal(first, second)
        self.assertEqual(assign_clusters(first).shape, (4,))

    def test_kmeans_finds_two_separated_groups(self):
        points = np.array([[0.0], [0.1], [5.0], [5.1]])
        labels, centers = kmeans(points, 2, seed=1)
        self.assertEqual(len(set(labels[:2])), 1)
        self.assertEqual(len(set(labels[2:])), 1)
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(centers.shape, (2, 1))

    def test_cli_operations_have_expected_output_shapes(self):
        points = np.array([[0.0], [1.0], [3.0]])
        self.assertEqual(len(run(2, "sym", points).splitlines()), 3)
        self.assertEqual(len(run(2, "symnmf", points).splitlines()), 3)
        self.assertEqual(len(run(2, "kmeans", points).splitlines()), 3)
        self.assertEqual(format_matrix([[1.23456]]), "1.2346")

    def test_invalid_cluster_count_is_rejected(self):
        with self.assertRaises(ValueError):
            kmeans([[0.0], [1.0]], 3)
        with self.assertRaises(ValueError):
            symnmf_factorization(np.zeros((2, 2)), 0)


if __name__ == "__main__":
    unittest.main()
