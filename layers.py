import numpy as np

X = [[1,2,3], [2.1,0.5, -0.4], [0.2, 2, -0.5]]

class denseLayer:
    weights = []
    def __init__(self, numIn, numNeu):

        self.weights = 0.10 * np.random.randn(numIn, numNeu)
        self.biases = np.zeros(numIn, numNeu)

    def forward(self, inputs):
        self.output = np.dot(inputs , self.weights) + self.biases




