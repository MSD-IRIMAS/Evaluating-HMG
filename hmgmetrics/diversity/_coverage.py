"""
This work was proposed by [1], this implementation is motivated from there code in
https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
Copyright (c) 2020-present NAVER Corp.
MIT license

[1] Naeem, Muhammad Ferjad, et al. "Reliable fidelity and diversity metrics for generative models."
    International Conference on Machine Learning. PMLR, 2020.
"""

from typing import *
import numpy as np
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from hmgmetrics.base import BaseMetricCalculator


class COVERAGE(BaseMetricCalculator):
    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
        n_neighbors: int = 5,
    ) -> None:
        self.n_neighbors = n_neighbors

        super(COVERAGE, self).__init__(classifier=classifier, batch_size=batch_size)

    def get_distances_k_neighbors(self, x: np.ndarray, k: int) -> np.ndarray:
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X=x)

        distances_neighbors, _ = nn.kneighbors(X=x)

        return distances_neighbors[:, k - 1]

    def calculate(
        self,
        xgenerated: np.ndarray = None,
        ygenerated: np.ndarray = None,
        xreal: np.ndarray = None,
        yreal: np.ndarray = None,
    ) -> float:
        if xreal is not None:
            if xgenerated is None:
                xgenerated, xreal = train_test_split(
                    xreal, stratify=yreal, test_size=0.5
                )

        real_latent = self.to_latent(x=xreal)
        gen_latent = self.to_latent(x=xgenerated)

        real_gen_distance_matrix = pairwise_distances(X=real_latent, Y=gen_latent)
        real_distances_k_neighbors = self.get_distances_k_neighbors(
            x=real_latent, k=self.n_neighbors + 1
        )

        distances_nearest_neighbor_real_to_gen = np.min(
            real_gen_distance_matrix, axis=1
        )

        exists_inside_neighborhood = (
            distances_nearest_neighbor_real_to_gen < real_distances_k_neighbors
        )

        coverage = np.mean(exists_inside_neighborhood)

        return coverage
