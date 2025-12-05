import unittest

from ml4am.datasets import BlockMatrixGenerator
from ml4am.optimal_clustering import OptimalClustering


class TestOptimalClustering(unittest.TestCase):

    def setUp(self) -> None:
        generator = BlockMatrixGenerator(number_factors=100,
                                         number_blocks=6,
                                         minimum_block_size=2,
                                         random_state=1,
                                         sigma_signal=1.0,
                                         sigma_noise=0.5)
        self.data = generator.generate()
        self.expected_number_of_clusters = 6
    def test_optimal_clustering(self):

        optimal_clustering_model = OptimalClustering(max_number_clusters=20, n_jobs=10)
        optimal_clustering_model.fit(self.data)

        self.assertAlmostEqual(
            self.expected_number_of_clusters,
            optimal_clustering_model.n_clusters

        )
