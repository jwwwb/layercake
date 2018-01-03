import numpy as np
import layercake as lc

class NeuronLayer(lc.Layer):
    """
    A single layer perceptron without any nonlinearity. Takes an arbitrary
    shaped tensor with trailing dimension `input_size` and transforms that
    trailing dimension to `output_size`
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.kernel = np.random.normal(0, 1.0/np.sqrt(input_size+output_size),
                                       input_size * output_size)
        self.kernel_shape = [input_size, output_size]
        self.bias = np.zeros(output_size)
        self.d_kernel = np.zeros_like(self.kernel)
        self.d_bias = np.zeros_like(self.bias)
        self.input_tensor = None

    def forward(self, input_tensor):
        # evaluates input and returns output
        # input_tensor is [batch_size, seq_len, input_size]
        # output is [batch_size, seq_len, output_size]
        self.input_tensor = input_tensor
        return np.dot(input_tensor, self.kernel.reshape(self.kernel_shape)) + self.bias

    def backward(self, gradient_tensor):
        # evaluates gradient at output, computes gradients for local updates
        # and returns gradient at input
        # gradient_tensor is [batch_size, seq_len, output_size]
        ax = np.arange(len(np.shape(gradient_tensor)))[:-1]
        self.d_bias[:] = np.sum(gradient_tensor, axis=tuple(ax))
        self.d_kernel[:] = np.tensordot(self.input_tensor, gradient_tensor,
                                     axes=[ax, ax]).reshape(-1)
        return np.dot(gradient_tensor, self.kernel.reshape(self.kernel_shape).T)

    #  def update(self, learning_rate):
    #      # updates weights according to the computed gradients from backward
    #      self.bias -= learning_rate * self.d_bias
    #      self.kernel -= learning_rate * self.d_kernel

    def get_gradients(self):
        return [self.d_kernel, self.d_bias]

    def get_weights(self):
        return [self.kernel, self.bias]

