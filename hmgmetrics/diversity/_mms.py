from typing import *
import numpy as np
import tensorflow as tf

from hmgmetrics.base import BaseMetricCalculator

from sklearn.neighbors import NearestNeighbors


class MMS(BaseMetricCalculator):
    def __init__(
        self, classifier: tf.keras.models.Model, batch_size: int = 128
    ) -> None:
        super(MMS, self).__init__(classifier=classifier, batch_size=batch_size)

    def calculate(
        self,
        xgenerated: np.ndarray = None,
        ygenerated: np.ndarray = None,
        xreal: np.ndarray = None,
        yreal: np.ndarray = None,
    ) -> float:
        k = 1
        if xgenerated is None:
            xgenerated = xreal
            k = 2

        real_latent = self.to_latent(x=xreal)
        gen_latent = self.to_latent(x=xgenerated)

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X=real_latent)

        distances, _ = nn.kneighbors(X=gen_latent, return_distance=True)

        if distances.shape[-1] > 1:
            print(distances.shape)
            distances = distances[:, 1]

        return np.mean(distances)
