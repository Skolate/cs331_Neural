import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


class MLP():
    def __init__(self, hidden_layer_count):
        self.model = Sequential()

        self.model.add(Dense(units=3, activation='sigmoid', input_dim=3, use_bias=False))
        self.model.add(Dense(units=hidden_layer_count, activation='sigmoid', use_bias=False))
        self.model.add(Dense(units=1, activation='sigmoid', use_bias=False))   

        self.model.compile(
            loss='mean_squared_error',
            optimizer=SGD(lr=1, momentum=0),
            metrics=['binary_accuracy']
        )

    def fit(self, X, Y):
        self.history = self.model.fit(X, Y, epochs=1000, batch_size=1, shuffle=True)
        print (self.model.predict(X))
        self.model.summary()
