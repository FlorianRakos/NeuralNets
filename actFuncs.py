import numpy as np


class ActivationReLu:

    @staticmethod
    def forward(inputs):
        return np.maximum(0, inputs)

