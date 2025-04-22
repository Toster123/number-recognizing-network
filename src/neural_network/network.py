# import keras
from .layers import *
# from keras.api.datasets import mnist


class SequentalNetwork():
    def __init__(self):
        self.__layers = []
        self.__layers.append(Convolution2DLayer((28, 28, 1), (3, 3), 2))
        self.__layers.append(MaxPooling2DLayer((26, 26, 2), (2, 2)))







        # (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32') / 255

        # y_train = keras.utils.to_categorical(y_train, 10)

        # print('Размерность x_train:', x_train[0])
        # print(self.__layers[0].shifts)
        # print(self.__layers[0].feedforward(x_train[0]))
        # print(self.__layers[1].feedforward(self.__layers[0].feedforward(x_train[0])))









        # print('Размерность x_train:', x_train.shape)
        # print('Размерность x_train:', x_train[0][10])
        # print('Размерность x_train:', x_train[1][10])
        # print('Размерность x_train:', y_train[0])

        # self.__layers.append(Convolution2DLayer((28, 28, 1), 8, (3, 3)))
        # self.__layers.append(Convolution2DLayer((26, 26, 16), 16, (3, 3)))
        # self.__layers.append(MaxPooling2DLayer((24, 24, 16), (2, 2)))
        # self.__layers.append(Convolution2DLayer((12, 12, 16), 8, (3, 3)))
        # self.__layers.append(Convolution2DLayer((10, 10, 8), 10, (3, 3)))
        # self.__layers.append(MaxPooling2DLayer((8, 8, 10), (2, 2)))
        #
        # self.__layers.append(FlattenLayer((4, 4, 10)))
        # self.__layers.append(DenseLayer((10, 10, 10), (2, 2)))


    def categorical_crossentropy(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def train(self, x_train, y_train):
        pass

    def backpropogate(self, x_train, y_train):
        pass

    def feedforward(self, values):
        for layer in self.__layers:
            values = layer.feedforward(values)

        return values
