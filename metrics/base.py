import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod


class BaseMetricCalculator(ABC):
    def __init__(self, classifier=None, batch_size=128, get_features=True):
        self.classifier = classifier
        if get_features:
            self.feature_extractor = self.remove_classification_layer(
                model=self.classifier
            )
        self.batch_size = batch_size

    def remove_classification_layer(self, model):
        input_model = model.input
        output_model = model.layers[-2].output

        return tf.keras.models.Model(inputs=input_model, outputs=output_model)

    def split_to_batches(self, x):
        n = int(x.shape[0])

        if self.batch_size > n:
            return [x]

        batches = []

        for i in range(0, n - self.batch_size + 1, self.batch_size):
            batches.append(x[i : i + self.batch_size])

        if n % self.batch_size > 0:
            batches.append(x[i + self.batch_size : n])

        return batches

    def rejoin_batches(self, x, n):
        m = len(x)
        d = x[0].shape[-1]

        x_rejoin = np.zeros(shape=(n, d))
        filled = 0

        for i in range(m):
            _stop = len(x[i])

            x_rejoin[filled : filled + _stop, :] = x[i]

            filled += len(x[i])

        return x_rejoin

    def to_latent(self, x):
        x_batches = self.split_to_batches(x=x)
        x_latent = []

        for batch in x_batches:
            x_latent.append(self.feature_extractor(batch, training=False))

        return self.rejoin_batches(x=x_latent, n=len(x))

    @abstractmethod
    def calculate(
        self,
        xgenerated=None,
        ygenerated=None,
        xreal=None,
        yreal=None,
    ): ...
