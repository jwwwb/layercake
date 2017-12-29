import numpy as np
import layercake as lc

class MultiLayerPerceptron:
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layers=(128,),
                 hidden_nonlinearities="tanh",
                 output_nonlinearity=None):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        if not isinstance(self.hidden_layers, (list, tuple)):
            self.hidden_layers = [self.hidden_layers]
        self.hidden_nonlinearities = hidden_nonlinearities
        if not isinstance(self.hidden_nonlinearities, (list, tuple)):
            self.hidden_nonlinearities = [self.hidden_nonlinearities] \
                * len(self.hidden_layers)
        if hidden_layers is None:
            self.hidden_layers = None
            self.hidden_nonlinearities = None
        self.output_nonlinearity = output_nonlinearity
        self.layers = self.build_network()

    def build_network(self):
        layers = []
        if self.hidden_layers is None:
            layers.append(lc.NeuronLayer(self.input_size, self.output_size))
            layers.append(lc.NonLinearity(self.output_nonlinearity))
        else:
            for i, o, n in zip([self.input_size] + list(self.hidden_layers),
                               list(self.hidden_layers) + [self.output_size],
                               list(self.hidden_nonlinearities) +\
                               [self.output_nonlinearity]):
                layers.append(lc.NeuronLayer(i, o))
                layers.append(lc.NonLinearity(n))
        return layers

    def forward(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, gradient_tensor):
        grad = gradient_tensor
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

