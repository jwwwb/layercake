import numpy as np
import layercake as lc

class Layer:
    """
    Abstract base class for any object that can represent a single layer or
    collection of layers in a trainable neural network.
    """
    def forward(self, input_tensor):
        raise NotImplementedError

    def backward(self, gradient_tensor):
        raise NotImplementedError

    def get_gradients(self):
        return []

    def get_weights(self):
        return []

