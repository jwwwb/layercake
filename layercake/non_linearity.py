import numpy as np
import layercake as lc

class NonLinearity(lc.Layer):
    """
    A nonlinearity to stack on top of NeuronLayers.
    """
    def __init__(self, nonlinearity=None):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_tensors = []

    def __call__(self, input_layer):
        self.input_tensors.append(None)
        return super().__call__(input_layer)

    def forward(self, for_input=0):
        # evaluates input and returns output
        input_tensor = super().forward(for_input)
        self.input_tensors[for_input] = input_tensor
        if self.nonlinearity is None or self.nonlinearity == "none":
            return input_tensor
        elif self.nonlinearity == "tanh":
            return np.tanh(input_tensor)
        elif self.nonlinearity == "relu":
            return np.maximum(0., input_tensor)
        else:
            raise NotImplementedError("No such nonlinearity known:" + self.nonlinearity)

    def backward(self, gradient_tensor, for_input=0):
        # evaluates gradient at output and returns gradient at input
        # return value is derivative of nonlinearity function times input
        inp = self.input_tensors[for_input]
        #  print("nonlin gradient:", gradient_tensor)
        if self.nonlinearity is None or self.nonlinearity == "none":
            super().backward(gradient_tensor, for_input=for_input)
        elif self.nonlinearity == "tanh":
            gradient_tensor = (1 - np.tanh(inp) **2) * gradient_tensor
            super().backward(gradient_tensor, for_input=for_input)
        elif self.nonlinearity == "relu":
            gradient_tensor = np.maximum(0., np.sign(inp)) * gradient_tensor
            super().backward(gradient_tensor, for_input=for_input)
        else:
            raise NotImplementedError("No such nonlinearity known:" +\
                self.nonlinearity)

    def get_output_size(self, for_input=0):
        return self.input_layers[for_input].output_size

    def update(self, learning_rate):
        # no trainable parameters here
        pass


def tester():
    layer = lc.InputLayer([2])
    layer.assign(np.asarray([3, -3]))
    layer = lc.NeuronLayer(2, 2)(layer)
    layer.bias = np.zeros(2)
    layer.kernel = np.asarray([1, 0, 0, 2])
    nonlin = NonLinearity("relu")(layer)
    out = nonlin()
    print("out:", out)

    print("nonlin", nonlin)
    grad = nonlin.backward([-1, 1])
    print("grad:", grad)


if __name__ == '__main__':
    tester()

