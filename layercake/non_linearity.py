import numpy as np
import layercake as lc

class NonLinearity(lc.Layer):
    """
    A nonlinearity to stack on top of NeuronLayers.
    """
    def __init__(self, nonlinearity=None):
        self.nonlinearity = nonlinearity
        self.input_tensor = None

    def forward(self, input_tensor):
        # evaluates input and returns output
        self.input_tensor = input_tensor
        if self.nonlinearity is None or self.nonlinearity == "none":
            return input_tensor
        elif self.nonlinearity == "tanh":
            return np.tanh(input_tensor)
        else:
            raise NotImplementedError("No such nonlinearity known:" + self.nonlinearity)

    def backward(self, gradient_tensor):
        # evaluates gradient at output and returns gradient at input
        if self.nonlinearity is None or self.nonlinearity == "none":
            return gradient_tensor
        elif self.nonlinearity == "tanh":
            #  print("grad", gradient_tensor)
            return (1 - self.input_tensor * self.input_tensor) * gradient_tensor
        else:
            raise NotImplementedError("No such nonlinearity known:" + self.nonlinearity)

    def update(self, learning_rate):
        # no trainable parameters here
        pass

