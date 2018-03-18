import numpy as np
import layercake as lc


class Sequential:
    def __init__(self, input_layer, layers=None, optimizer=None):
        self.input_layer = input_layer
        self.layers = [] if layers is None else layers
        network = self.input_layer
        for layer in self.layers:
            network = layer(network)
        self.network = network
        if optimizer is None:
            self.optimizer = lc.Optimizer(0.01)
        else:
            self.optimizer = optimizer
        self.train_losses = []
        self.tl = []
        self.val_losses = []

    def add(self, layer):
        self.layers.append(layer)
        self.network = layer(self.network)

    def add_loss(self, loss_layer):
        self.target_layer = lc.InputLayer([None, self.network.output_size])
        self.loss_layer = loss_layer
        self.loss = self.loss_layer(self.target_layer, self.network)

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        for layer in self.layers:
            self.optimizer.add_parameters(layer.get_weights(), layer.get_gradients())

    def compile(self, optimizer, loss_layer):
        self.add_loss(loss_layer)
        self.add_optimizer(optimizer)

    def train_on_batch(self, observation, target):
        self.input_layer.assign(observation)
        self.target_layer.assign(target)
        loss = self.loss()
        self.loss.backward()
        self.optimizer.update()
        return loss

    def test_on_batch(self, observation, target):
        self.input_layer.assign(observation)
        self.target_layer.assign(target)
        loss = self.loss()
        pred = self.network()
        print(loss.shape, observation.shape, target.shape, pred.shape)
        print("test loss:", loss)
        print("obs, target, prediction")
        print(np.hstack([observation, target, pred]))
        return loss

    def predict_on_batch(self, observation):
        self.input_layer.assign(observation)
        prediction = self.network()
        return prediction

    def train(self, data_source, with_validation=None, num_epochs=10, tolerance=1.1):
        if with_validation is not None:
            self.train_with_validation(data_source, with_validation,
                                       min_epochs=num_epochs, tolerance=tolerance)
        elif num_epochs is not None:
            self.train_for_epochs(data_source, num_epochs=num_epochs)

    def train_with_validation(self, data_source, with_validation,
                            min_epochs=2, tolerance=1.1):
        e = 0
        keep_training = True
        best_loss = None
        while keep_training:
            loss = []
            tl = []
            for d in data_source:
                tl.append(np.mean(self.test_on_batch(d['X'], d['y'])))
                loss.append(np.mean(self.train_on_batch(d['X'], d['y'])))
            self.tl.extend(tl)
            self.train_losses.extend(loss)


            loss = []
            e += 1
            for v in with_validation:
                loss.append(np.mean(self.test_on_batch(v['X'], d['y'])))
            self.val_losses.extend(loss)
            loss = np.mean(loss)
            print("epoch {} loss: {}".format(e, loss))
            print("len_losses:", len(self.val_losses))
            keep_training = e < min_epochs or loss < best_loss * tolerance
            best_loss = loss if best_loss is None else min(loss, best_loss)

    def train_for_epochs(self, data_source, num_epochs):
        for e in range(num_epochs):
            for d in data_source:
                l = self.train_on_batch(d['X'], d['y'])
                self.losses.append(np.mean(l))


def tester():
    pass


if __name__ == '__main__':
    tester()

