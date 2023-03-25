import numpy as np
import actFuncs as act
import loss

X = np.array([[1,2,3], [2.1,0.5, -0.4], [0.2, 2, -0.5]])
Y = np.array([0,2,1])

class DenseLayer:
    def __init__(self, numIn, numNeu):
        self.weights = 0.10 * np.random.randn(numIn, numNeu)
        self.biases = np.zeros((1, numNeu))

    def forward(self, inputs):
        self.output = np.dot(inputs , self.weights) + self.biases


layer = DenseLayer(3, 5)

layer.forward(X)
output1 = act.ActivationReLu.forward(layer.output)
print (output1)

layer2 = DenseLayer(5,3)
layer2.forward(output1)
output2 = act.ActivationSoftmax.forward(layer2.output)

loss_function = loss.LossCategoricalCrossentropy()
loss = loss_function.calculate(output2, Y)
print(loss)




