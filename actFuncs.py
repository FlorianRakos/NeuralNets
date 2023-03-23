import numpy as np
import math


class ActivationReLu:

    @staticmethod
    def forward(inputs):
        return np.maximum(0, inputs)

class ActivationSoftmax:

    @staticmethod
    def forward(inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
