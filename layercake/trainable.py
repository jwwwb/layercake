import numpy as np
import layercake as lc

class Trainable:
    def __init__(self, input_layers, network, optimizer):
        self.input_layers = input_layers
        self.network = network
        self.optimizer = optimizer
        self.optimizer.add_parameters(self.network.get_weights(),
                                      self.network.get_gradients())
        self.losses = []

    def training_step(self, inputs):
        loss = self.evaluate(inputs)
        self.network.backward(None)
        self.optimizer.update()
        return loss

    def evaluate(self, inputs):
        for input_id in inputs:
            self.input_layers[input_id].assign(inputs[input_id])
        return self.network.forward(None)

    def train(self, data_source, num_epochs):
        for epoch in range(num_epochs):
            inputs = data_source()
            loss = self.training_step(inputs)
            self.losses.append(loss)
            for w in self.network.get_weights():
                print("min", np.min(w), "mean", np.mean(w),
                      "max", np.max(w), "shape", w.shape)
            print("epoch {}. loss = {}".format(epoch, loss))
            print("*"*30)

