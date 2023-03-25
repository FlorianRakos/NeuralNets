import math
import numpy as np



class Loss:


    def calculate(self, output, y):
        losses = self.forward(output, y)
        return np.mean(losses)


class LossCategoricalCrossentropy(Loss):
    def forward(self, pred, labels):
        samples = len(pred)

        # clip, because log(0) = inf
        y_pred_clipped = np.clip(pred, 1e-7, 1 - 1e-7)

        # Index encoded
        if len (labels.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), labels]

        # One Hot encoded
        elif len(labels.shape) == 2:
            # sum removes 0 values
            correct_confidences = np.sum(y_pred_clipped * labels, axis=1)

        return -np.log(correct_confidences)


# data = np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
# labels = [2, 1]
#
# Loss.calculate(data, labels)


