import numpy as np
import layercake as lc


class Output:
    """
    An output always returns the same value during a session. Outputs can
    either represent placeholders / input layers or be the connection between
    two full layers.
    """
    def __init__(self, layer, for_input=0):
        assert isinstance(layer, lc.Layer)
        self.layer = layer
        self.for_input = for_input

    def __call__(self):
        return self.forward()

    def forward(self):
        return self.layer.forward(self.for_input)

    def backward(self, gradient_tensor=None):
        #  print("output gradient:", gradient_tensor)
        self.layer.backward(gradient_tensor, self.for_input)

    @property
    def output_size(self):
        return self.layer.get_output_size(self.for_input)


class Layer:
    """
    Abstract base class for any object that can represent a single layer or
    collection of layers in a trainable neural network.
    Layers can be reused by calling them on different inputs, thus pruducing
    different outputs.
    """
    def __init__(self):
        self.input_layers = []
        self.outputs = []
        self.output_layers = []

    def __call__(self, input_layer):
        assert isinstance(input_layer, lc.Output)
        output = Output(self, len(self.input_layers))
        self.input_layers.append(input_layer)
        self.outputs.append(output)
        return output

    def forward(self, for_input=0):
        return self.input_layers[for_input].forward()

    def backward(self, gradient_tensor, for_input=0):
        #  print("layer gradient:", gradient_tensor)
        self.input_layers[for_input].backward(gradient_tensor)

    def get_output_size(self, for_input=0):
        raise NotImplementedError

    def get_gradients(self):
        return []

    def get_weights(self):
        return []


class MultipleInputLayer(Layer):
    def __call__(self, input_layer):
        assert isinstance(input_layer, (list, tuple))
        for layer in input_layer:
            assert isinstance(layer, lc.Output)
        output = Output(self, len(self.input_layers))
        self.input_layers.append(input_layer)
        self.outputs.append(output)
        return output

    def forward(self, for_input=0):
        return [layer.forward() for layer in self.input_layers[for_input]]

    def backward(self, gradient_tensor, for_input=0):
        for gradient, layer in zip(gradient_tensor, self.input_layers[for_input]):
            layer.backward(gradient)

if __name__ == '__main__':
    pass

