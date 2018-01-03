import numpy as np
import layercake as lc

class LossLayer(lc.Layer):
    def __init__(self, loss):
        self.loss = loss
        self.target = None
        self.input_tensor = None

    def forward(self, input_tensor):
        assert self.target is not None
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - self.target))

    def backward(self, gradient_tensor=None):
        assert gradient_tensor is None
        return 2 * (self.input_tensor - self.target)

    def update(self, learning_rate):
        # no trainable parameters here
        pass

    def assign(self, target):
        self.target = target

class CategoricalLossLayer(LossLayer):
    def forward(self, input_tensor):
        assert self.target is not None
        softmax = np.exp(input_tensor) / np.sum(np.exp(input_tensor), axis=-1,
                                                keepdims=True)
        softmax_flat = softmax.reshape(-1, np.shape(softmax)[-1])
        target_flat = self.target.reshape(-1)
        loss = np.sum(np.log(softmax_flat[np.arange(len(target_flat)),
                                                    target_flat]))
        self.input_tensor = softmax
        return loss

    def backward(self, gradient_tensor=None):
        assert gradient_tensor is None
        gradient = np.copy(self.input_tensor)
        adder = np.zeros_like(gradient).reshape(-1, np.shape(gradient)[-1])
        target_flat = self.target.reshape(-1)
        adder[np.arange(len(target_flat)), target_flat] += 1
        return gradient - adder.reshape(np.shape(gradient))


