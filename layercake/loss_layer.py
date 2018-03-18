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
        return self.input_layers[for_input][0].output_size

    #  def update(self, learning_rate):
    #      # no trainable parameters here
    #      pass

    #  def assign(self, target):
    #      self.target = target

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


