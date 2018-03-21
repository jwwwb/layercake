import numpy as np
import layercake as lc
from pprint import pprint

class Optimizer:
    """
    Optimizer has a collection of pointers to all trainable parameters as
    well as their gradients. In the default version it simply performs SGD.
    """
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weights = []
        self.gradients = []

    def add_parameters(self, weights, gradients):
        self.weights.extend(weights)
        self.gradients.extend(gradients)

    def update(self):
        #  print("weights:")
        #  pprint(self.weights)
        #  print()
        #  print("gradients:")
        #  pprint(self.gradients)
        #  print()
        for weight, grad in zip(self.weights, self.gradients):
            #  print('updating', weight, 'by', grad)
            weight[:] -= self.learning_rate * grad


class L2Optimizer(Optimizer):
    def update(self):
        for weight, grad in zip(self.weights, self.gradients):
            weight[:] -= self.learning_rate * grad
            weight[:] *= 0.9


class MomentumOptimizer(Optimizer):
    """
    Maintains a tensor of the previous update and uses momentum to smooth
    the SGD updates.
    """
    def __init__(self, momentum, **kwargs):
        self.momentum = momentum
        self.previous_updates = []
        super().__init__(**kwargs)

    def add_parameters(self, weights, gradients):
        super().add_parameters(weights, gradients)
        for gradient in gradients:
            self.previous_updates.append(np.zeros_like(gradient))

    def update(self):
        for weight, grad, previous in zip(self.weights,
                                          self.gradients,
                                          self.previous_updates):
            previous *= self.momentum
            previous += self.learning_rate * grad
            weight[:] -= previous


class NesterovOptimizer(Optimizer):
    pass
