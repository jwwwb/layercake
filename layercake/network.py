import numpy as np
import layercake as lc

class Network:
    def __init__(self, layers=None):
        if isinstance(layers, (list, tuple)):
            self.layers = list(layers)
        else:
            self.layers = []

    def append(self, layers):
        if isinstance(layers, (list, tuple)):
            self.layers.extend(list(layers))
        else:
            self.layers.append(layers)

    def forward(self, input_tensor=None):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def backward(self, gradient_tensor=None):
        for layer in reversed(self.layers):
            gradient_tensor = layer.backward(gradient_tensor)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

