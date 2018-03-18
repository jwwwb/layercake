import layercake as lc
import numpy as np
import tensorflow as tf
import keras

def train_model(model=None):
    np.random.seed(2)
    X = np.random.normal(0, 1, [5, 2])
    y = np.sum(X**2, axis=-1, keepdims=True)
    data_set = {'X': X, 'y': y}
    train, val = [lc.DataSource(d, 1) for d in lc.split_data(data_set, [.8, .2])]
    for _ in range(100):
        for v in val:
            print(np.mean(model.test_on_batch(v['X'], v['y'])))
        for t in train:
            model.train_on_batch(t['X'], t['y'])
    for v in val:
        print(np.mean(model.test_on_batch(v['X'], v['y'])))


def build_keras_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(16, input_dim=2))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01,
                                                 momentum=0.0,
                                                 decay=0.0,
                                                 nesterov=False),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model

def build_lc_model():
    input_layer = lc.InputLayer(2)
    model = lc.Sequential(input_layer)
    model.add(lc.NeuronLayer(16))
    model.add(lc.NonLinearity('relu'))
    model.add(lc.NeuronLayer(1))
    model.add_loss(lc.LossLayer())
    model.add_optimizer(lc.Optimizer(0.01))

    return model

if __name__ == '__main__':
    model = build_keras_model()
    lcmodel = build_lc_model()
    print("keras")
    train_model(model)
    print("layercake")
    train_model(lcmodel)

