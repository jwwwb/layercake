import numpy as np
import layercake as lc

class NeuronLayer(lc.Layer):
    """
    A single layer perceptron without any nonlinearity. Takes an arbitrary
    shaped tensor with trailing dimension `input_size` and transforms that
    trailing dimension to `output_size`
    """
    def __init__(self, output_size, input_size=None):
        super().__init__()
        self.input_tensors = []
        self.output_size = output_size
        self.input_size = None
        if input_size is not None:
            self.set_input_size(input_size)

    def set_input_size(self, input_size):
        if self.input_size is not None:
            if input_size != self.input_size:
                raise AttributeError(
                    "Cannot connect neuron layer to different size inputs.")
        else:
            self.input_size = input_size
            limit = np.sqrt(6. / (input_size+self.output_size))
            self.kernel = np.random.uniform(-limit, limit,
                                            input_size * self.output_size)
            #  self.kernel = np.zeros(input_size * self.output_size)
            self.kernel_shape = [input_size, self.output_size]
            self.bias = np.zeros(self.output_size)
            self.d_kernel = np.zeros_like(self.kernel)
            self.d_bias = np.zeros_like(self.bias)
            self.input_tensor = None

    def __call__(self, input_layer):
        self.input_tensors.append(None)
        self.set_input_size(input_layer.output_size)
        return super().__call__(input_layer)

    def forward(self, for_input=0):
        # gets input from previous layer and returns output
        # input_tensor is [batch_size, seq_len, input_size]
        # output is [batch_size, seq_len, output_size]
        input_tensor = super().forward(for_input)
        self.input_tensors[for_input] = input_tensor
        return np.dot(input_tensor,
                      self.kernel.reshape(self.kernel_shape)) + self.bias

    def backward(self, gradient_tensor, for_input=0):
        # evaluates gradient at output, computes gradients for local updates
        # and returns gradient at input
        # gradient_tensor is [batch_size, seq_len, output_size]
        inp = self.input_tensors[for_input]
        #  print("neuronlayer gradient:", gradient_tensor)
        ax = np.arange(len(np.shape(gradient_tensor)))[:-1]
        self.d_bias[:] = np.sum(gradient_tensor, axis=tuple(ax))
        self.d_kernel[:] = np.tensordot(inp, gradient_tensor,
                                     axes=[ax, ax]).reshape(-1)
        #  print("d_kernel:", self.d_kernel)
        #  print("d_bias:", self.d_bias)
        gradient_tensor = np.dot(gradient_tensor,
                                 self.kernel.reshape(self.kernel_shape).T)
        super().backward(gradient_tensor, for_input=for_input)

    #  def update(self, learning_rate):
    #      # updates weights according to the computed gradients from backward
    #      self.bias -= learning_rate * self.d_bias
    #      self.kernel -= learning_rate * self.d_kernel

    def get_output_size(self, for_input=0):
        return self.output_size

    def get_gradients(self):
        return [self.d_kernel, self.d_bias]

    def get_weights(self):
        return [self.kernel, self.bias]

