from typing import *
import numpy as np
import tensorflow as tf

from hmgmetrics.base import BaseMetricCalculator


class ACPD(BaseMetricCalculator):
    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
        Sapd: int = 200,
        runs: int = 5,
    ) -> None:
        self.Sapd = Sapd
        self.runs = runs

        super(ACPD, self).__init__(classifier=classifier, batch_size=batch_size)

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
                ygenerated = yreal

        self.n_classes = len(np.unique(ygenerated))
        acpd_values = []

        for c in range(self.n_classes):
            apd_per_class_values = []
            x_c = xgenerated[ygenerated == c]

            x_c_latent = self.to_latent(x=x_c)

            if self.Sapd > len(x_c):
                self._Sapd = len(x_c)
            else:
                self._Sapd = self.Sapd

            for _ in range(self.runs):
                all_indices = np.arange(len(x_c))

                V = x_c_latent[np.random.choice(a=all_indices, size=self._Sapd)]
                V_prime = x_c_latent[np.random.choice(a=all_indices, size=self._Sapd)]

                apd_per_class_values.append(
                    np.mean(np.linalg.norm(V - V_prime, axis=1))
                )

            acpd_values.append(np.mean(apd_per_class_values))

        return np.mean(acpd_values)
