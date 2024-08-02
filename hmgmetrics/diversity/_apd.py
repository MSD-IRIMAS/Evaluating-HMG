from typing import *
import numpy as np
import tensorflow as tf

from hmgmetrics.base import BaseMetricCalculator


class APD(BaseMetricCalculator):
    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
        Sapd: int = 200,
        runs: int = 5,
    ) -> None:
        self.Sapd = Sapd
        self.runs = runs

        super(APD, self).__init__(classifier=classifier, batch_size=batch_size)

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

        x_latent = self.to_latent(x=xgenerated)

        if self.Sapd > len(xgenerated):
            self._Sapd = len(xgenerated)
        else:
            self._Sapd = self.Sapd

        apd_values = []

        for _ in range(self.runs):
            all_indices = np.arange(len(xgenerated))

            V = x_latent[np.random.choice(a=all_indices, size=self._Sapd)]
            V_prime = x_latent[np.random.choice(a=all_indices, size=self._Sapd)]

            apd_values.append(np.mean(np.linalg.norm(V - V_prime, axis=1)))

        return np.mean(apd_values)
