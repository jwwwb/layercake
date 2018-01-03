import numpy as np
import layercake as lc

class InputLayer(lc.Layer):
    def __init__(self):
        self.value = None

    def forward(self, input_tensor=None):
        # provides the value assigned to the layer to start off a network
        assert input_tensor is None
        assert self.value is not None
        return self.value

    def backward(self, gradient_tensor):
        # end of back propagation chain
        return None

    def update(self, learning_rate):
        # no trainable parameters here
        pass

    def assign(self, value):
        self.value = value

