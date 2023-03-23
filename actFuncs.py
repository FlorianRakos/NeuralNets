import numpy as np


class ActivationReLu:

    @staticmethod
    def forward(inputs):
        return np.max(0, inputs)