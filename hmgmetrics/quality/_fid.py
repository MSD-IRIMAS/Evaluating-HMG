from typing import *
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm

from hmgmetrics.base import BaseMetricCalculator
from sklearn.model_selection import train_test_split


class FID(BaseMetricCalculator):
    def __init__(
        self, classifier: tf.keras.models.Model, batch_size: int = 128
    ) -> None:
        super(FID, self).__init__(classifier=classifier, batch_size=batch_size)

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

        mean_real = np.mean(real_latent, axis=0)
        cov_real = np.cov(real_latent, rowvar=False)

        gen_latent = self.to_latent(x=xgenerated)

        mean_gen = np.mean(gen_latent, axis=0)
        cov_gen = np.cov(gen_latent, rowvar=False)

        diff_means = np.sum(np.square(mean_real - mean_gen))
        cov_prod = sqrtm(cov_real.dot(cov_gen))

        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real

        fid = diff_means + np.trace(cov_real + cov_gen - 2.0 * cov_prod)

        return fid
