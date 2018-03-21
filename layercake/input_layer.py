import numpy as np
import layercake as lc


def stats(tensor):
    print("min:", np.min(tensor), "\tmean:", np.mean(tensor), "\tmax:", np.max(tensor))


class InputLayer(lc.Output):
    def __init__(self, shape):
        self.shape = [None, shape] if isinstance(shape, int) else shape
        self.value = None

    def __call__(self, input_layer):
        raise AttributeError("cannot connect input layer to other layer")

    def forward(self):
        # provides the value assigned to the layer to start off a network
        assert self.value is not None
        #  print(self.__class__.__name__, self.value.shape)
        #  stats(self.value)
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
        if value.shape[-1] != self.shape[-1]:
            print("value:", value.shape, "self:", self.shape)
            raise ValueError("Shapes of assignment must match.")
        self.value = value

