"""
 Deep Neural Network model
"""

import keras
from keras.layers import Input, Conv2D, Flatten, MaxPooling2D, Dense

from dependencies.utilities import Timer

class DNNmodel():

    def __init__(self, env, seed=None):

        initializer = keras.initializers.he_uniform(seed)

        inputs = Input(shape=(env.observation_space.shape))

        # Convolutions on the frames on the screen
        layer1 = Conv2D(32, 8, strides=4, activation="relu", kernel_initializer=initializer)(inputs)
        layer2 = Conv2D(64, 4, strides=2, activation="relu", kernel_initializer=initializer)(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu", kernel_initializer=initializer)(layer2)

        layer4 = MaxPooling2D(pool_size=(1, 1))(layer3)

        layer5 = Flatten()(layer4)

        layer6 = Dense(512, activation="relu", kernel_initializer=initializer)(layer5)
        action = Dense(env.action_space.n, activation="linear", kernel_initializer=initializer)(layer6)

        self.kerasModel = keras.Model(inputs=inputs, outputs=action)

    def get_weights(self):
        return self.kerasModel.get_weights()

    def set_weights(self, weights):
        self.kerasModel.set_weights(weights)

    def save_model(self, path):
        self.kerasModel.save(path)

    def load_model(self, path):
        self.kerasModel = keras.models.load_model(path)

    @Timer(name="DNN-predict")
    def predict(self, input):
        return self.kerasModel.predict(input, batch_size=1)[0]
