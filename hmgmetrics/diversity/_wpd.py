from typing import *
import numpy as np
import tensorflow as tf

from aeon.distances import dtw_alignment_path
from hmgmetrics.base import BaseMetricCalculator


class WPD(BaseMetricCalculator):
    def __init__(
        self,
        classifier: tf.keras.models.Model = None,
        batch_size: int = 128,
        Swpd: int = 200,
        runs: int = 5,
    ) -> None:
        self.Swpd = Swpd
        self.runs = runs

        super(WPD, self).__init__(
            classifier=classifier, batch_size=batch_size, get_features=False
        )

    def calculate(
        self,
        xgenerated: np.ndarray = None,
        ygenerated: np.ndarray = None,
        xreal: np.ndarray = None,
        yreal: np.ndarray = None,
    ) -> float:
        if xreal is not None:
            if xgenerated is None:
                xgenerated = xreal

        if self.Swpd > len(xgenerated):
            self._Swpd = len(xgenerated)
        else:
            self._Swpd = self.Swpd

        wpd_values = []

        if len(xgenerated.shape) > 3:
            xgenerated = np.reshape(
                xgenerated, (xgenerated.shape[0], xgenerated.shape[1], -1)
            )

        for _ in range(self.runs):
            all_indices = np.arange(len(xgenerated))

            G = xgenerated[np.random.choice(a=all_indices, size=self._Swpd)]
            G_prime = xgenerated[np.random.choice(a=all_indices, size=self._Swpd)]

            for i in range(self._Swpd):
                dtw_path, dtw_dist = dtw_alignment_path(x=G[i], y=G_prime[i])
                dtw_path = np.asarray(dtw_path)

                wpd_values.append(
                    (np.sqrt(2) / (2 * len(dtw_path)))
                    * np.sum(np.abs(dtw_path[:, 0] - dtw_path[:, 1]))
                )

        return np.mean(wpd_values)
