import numpy as np
import layercake as lc

class Trainable:
    def __init__(self):
        self.input_layers = {}
        self.network = None

    def training_step(self, inputs, learning_rate):
        loss = self.evaluate(inputs)
        self.network.backward(None)
        self.network.update(learning_rate)
        return loss

    def evaluate(self, inputs):
        for input_id in inputs:
            self.input_layers[input_id].assign(inputs[input_id])
        return self.network.forward(None)

    def train(self, data_source, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            inputs = data_source()
            loss = self.training_step(inputs, learning_rate)
            print("epoch {}. loss = {}".format(epoch, loss))

