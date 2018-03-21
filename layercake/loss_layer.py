import numpy as np
import layercake as lc

class LossLayer(lc.MultipleInputLayer):
    def __init__(self):
        super().__init__()
        self.input_tensors = []

    def __call__(self, targets, inputs):
        assert targets.output_size == inputs.output_size
        self.input_tensors.append(None)
        return super().__call__([targets, inputs])

    def forward(self, for_input=0):
        target, input_tensor = super().forward(for_input)
        self.input_tensors[for_input] = (target, input_tensor)
        return np.mean(np.square(input_tensor - target), axis=-1)

    def backward(self, gradient_tensor=None, for_input=0):
        assert gradient_tensor is None
        target, input_tensor = self.input_tensors[for_input]
        gradient = 2 * (target - input_tensor)
        #  print("gradient before:", gradient)
        output_size = self.get_output_size(for_input)
        #  print("output_size:", output_size)
        gradient /= output_size
        #  print("gradient after:", gradient)
        super().backward((gradient, - gradient), for_input=for_input)

    def get_output_size(self, for_input=0):
        return (1)
        #  return self.input_layers[for_input][0].output_size

    #  def update(self, learning_rate):
    #      # no trainable parameters here
    #      pass

    #  def assign(self, target):
    #      self.target = target

def stats(tensor):
    pass
    #  print("min:", np.min(tensor), "\tmean:", np.mean(tensor), "\tmax:", np.max(tensor))
def print(*args):
    pass


class CategoricalLossLayer(lc.MultipleInputLayer):
    def __init__(self):
        super().__init__()
        self.input_tensors = []

    def __call__(self, targets, inputs):
        self.input_tensors.append(None)
        return super().__call__([targets, inputs])

    def forward(self, for_input=0):
        target, input_tensor = super().forward(for_input)
        self.input_tensors[for_input] = (target, input_tensor)
        print("input_tensor", input_tensor.shape)
        stats(input_tensor)
        activation = np.exp(input_tensor)
        softmax = activation / np.sum(activation, axis=-1, keepdims=True)
        softmax_flat = softmax.reshape(-1, np.shape(softmax)[-1])
        target_flat = target.reshape(-1)
        #  print("softmax_flat", softmax_flat.shape)
        #  print(softmax_flat)
        print("target_flat", target_flat.shape)
        stats(target_flat)
        exp_loss = softmax_flat[np.arange(len(target_flat)), target_flat]
        print("exp_loss", exp_loss.shape)
        stats(exp_loss)
        loss_tensor = - np.log(exp_loss)
        print("loss_tensor", loss_tensor.shape)
        stats(loss_tensor)
        #  print("loss_tensor", loss_tensor.shape)
        #  print(loss_tensor)
        loss = np.sum(loss_tensor)
        return loss

    def backward(self, gradient_tensor=None, for_input=0):
        assert gradient_tensor is None
        target, input_tensor = self.input_tensors[for_input]
        activation = np.exp(input_tensor)
        softmax = activation / np.sum(activation, axis=-1, keepdims=True)
        gradient = np.copy(softmax)
        print('\t\tgradient', gradient.shape)
        stats(gradient)
        adder = np.zeros_like(gradient).reshape(-1, np.shape(gradient)[-1])
        target_flat = target.reshape(-1)
        adder[np.arange(len(target_flat)), target_flat] += 1
        print("\t\tadder", adder.shape)
        stats(adder)
        gradient_tensor = gradient - adder.reshape(np.shape(gradient))
        print("\t\tGT", gradient_tensor.shape)
        stats(gradient_tensor)
        super().backward([None, gradient_tensor], for_input=for_input)

    def get_output_size(self):
        return (1)
