import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


class MLP():
    def __init__(self, hidden_layer_count):
        self.model = Sequential()

        self.model.add(Dense(units=3, activation='linear', input_dim=3))
        self.model.add(Dense(units=hidden_layer_count, activation='linear'))
        self.model.add(Dense(units=1, activation='sigmoid'))   

        self.model.compile(
            loss='mse',
            optimizer=SGD(lr=0.02, momentum=0),
            metrics=['accuracy']
        )

    def fit(self, X, Y):
        self.model.fit(X, Y, epochs=1000)