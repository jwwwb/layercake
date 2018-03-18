import numpy as np
import layercake as lc

class InputLayer(lc.Output):
    def __init__(self, shape):
        self.shape = [None, shape] if isinstance(shape, int) else shape
        self.value = None

    def __call__(self, input_layer):
        raise AttributeError("cannot connect input layer to other layer")

    def forward(self):
        # provides the value assigned to the layer to start off a network
        assert self.value is not None
        return self.value

    def backward(self, gradient_tensor):
        # end of back propagation chain
        #  print(gradient_tensor)
        return

    @property
    def output_size(self):
        return self.shape[-1]

    def update(self, learning_rate):
        # no trainable parameters here
        pass

    def assign(self, value):
        assert len(value.shape) == len(self.shape)
        assert value.shape[-1] == self.shape[-1]
        self.value = value

