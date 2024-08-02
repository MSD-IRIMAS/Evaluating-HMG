from typing import *
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score

from hmgmetrics.base import BaseMetricCalculator


class AOG(BaseMetricCalculator):
    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
    ) -> None:
        super(AOG, self).__init__(
            classifier=classifier, batch_size=batch_size, get_features=False
        )

    def calculate(
        self,
        xgenerated: np.ndarray = None,
        ygenerated: np.ndarray = None,
        xreal: np.ndarray = None,
        yreal: np.ndarray = None,
    ) -> float:
        if xreal is not None and yreal is not None:
            if xgenerated is None and ygenerated is None:
                xgenerated = xreal
                ygenerated = yreal

        ypred = self.classifier.predict(xgenerated, batch_size=self.batch_size)
        ypred = np.argmax(ypred, axis=1)

        return accuracy_score(y_true=ygenerated, y_pred=ypred, normalize=True)
