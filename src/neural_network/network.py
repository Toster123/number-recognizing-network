import os
from keras.api.datasets import mnist
from keras.api.models import Sequential
from keras.api.layers import Dense, Flatten
from keras.api.layers import Conv2D, MaxPooling2D
from .layers import *


class SequentalNetwork():
    def __init__(self):
        if not (os.path.exists('weights.txt') and (f:=open('weights.txt', 'r'))):
            f=0
        self.init_weights(f)

    def init_weights(self, f=0):
        kernels1 = 0
        shifts1 = 0
        kernels2 = 0
        shifts2 = 0
        kernels3 = 0
        shifts3 = 0
        kernels4 = 0
        shifts4 = 0
        weights5 = 0
        shifts5 = 0
        weights6 = 0
        shifts6 = 0
        weights7 = 0
        shifts7 = 0

        if f:
            weights = np.array(list(map(np.float64, f.readlines())))

            if len(weights) >= 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 + 128 + 128*10 + 10:
                kernels1 = weights[:32*3*3].reshape((32, 1, 3, 3))
                shifts1 = weights[32*3*3 : 32*3*3 + 32].reshape((32))

                kernels2 = weights[32*3*3 + 32 : 32*3*3 + 32 + 32*32*3*3].reshape((32, 32, 3, 3))
                shifts2 = weights[32*3*3 + 32 + 32*32*3*3 : 32*3*3 + 32 + 32*32*3*3 + 32].reshape((32))

                kernels3 = weights[32*3*3 + 32 + 32*32*3*3 + 32 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3].reshape((64, 32, 3, 3))
                shifts3 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64].reshape((64))

                kernels4 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3].reshape((64, 64, 3, 3))
                shifts4 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64].reshape((64))

                weights5 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64].reshape((256, 64*4*4))
                shifts5 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256].reshape((256))

                weights6 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256].reshape((128, 256))
                shifts6 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 + 128].reshape((128))

                weights7 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 + 128 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 + 128 + 128*10].reshape((10, 128))
                shifts7 = weights[32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 + 128 + 128*10 : 32*3*3 + 32 + 32*32*3*3 + 32 + 64*32*3*3 + 64 + 64*64*3*3 + 64 + 256*4*4*64 + 256 + 128*256 + 128 + 128*10 + 10].reshape((10))

        self.__layers = []
        self.__layers.append(Convolution2DLayer((1, 28, 28), (3, 3), 32, kernels1, shifts1))
        self.__layers.append(Convolution2DLayer((32, 26, 26), (3, 3), 32, kernels2, shifts2))
        self.__layers.append(MaxPooling2DLayer((32, 24, 24), (2, 2)))
        self.__layers.append(Convolution2DLayer((32, 12, 12), (3, 3), 64, kernels3, shifts3))
        self.__layers.append(Convolution2DLayer((64, 10, 10), (3, 3), 64, kernels4, shifts4))
        self.__layers.append(MaxPooling2DLayer((64, 8, 8), (2, 2)))
        self.__layers.append(FlattenLayer((64, 8, 8)))
        self.__layers.append(DenseLayer(64*4*4, 256, weights5, shifts5))
        self.__layers.append(DenseLayer(256, 128, weights6, shifts6))
        self.__layers.append(DenseLayer(128, 10, weights7, shifts7, 'softmax'))

    def train(self):
        try:
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

            y_train = keras.utils.to_categorical(y_train, 10)
            y_test = keras.utils.to_categorical(y_test, 10)

            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(10, activation='softmax'))

            model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            model.fit(x_train, y_train, batch_size=64, epochs=1, verbose=1, validation_data=(x_test, y_test))

        except:
            raise Exception('Ошибка во время обучения')

        try:
            f=open('weights.txt', 'w')

            f.writelines(list(map(to_string, model.layers[0].get_weights()[0].reshape((3, 3, 1, 32)).transpose((3, 2, 0, 1)).flatten())))
            f.writelines(list(map(to_string, model.layers[0].get_weights()[1].reshape((32)))))
            f.writelines(list(map(to_string, model.layers[1].get_weights()[0].reshape((3, 3, 32, 32)).transpose((3, 2, 0, 1)).flatten())))
            f.writelines(list(map(to_string, model.layers[1].get_weights()[1].reshape((32)))))
            f.writelines(list(map(to_string, model.layers[3].get_weights()[0].reshape((3, 3, 32, 64)).transpose((3, 2, 0, 1)).flatten())))
            f.writelines(list(map(to_string, model.layers[3].get_weights()[1].reshape((64)))))
            f.writelines(list(map(to_string, model.layers[4].get_weights()[0].reshape((3, 3, 64, 64)).transpose((3, 2, 0, 1)).flatten())))
            f.writelines(list(map(to_string, model.layers[4].get_weights()[1].reshape((64)))))
            f.writelines(list(map(to_string, model.layers[7].get_weights()[0].reshape((4 * 4 * 64, 256)).transpose((1, 0)).flatten())))
            f.writelines(list(map(to_string, model.layers[7].get_weights()[1].reshape((256)))))
            f.writelines(list(map(to_string, model.layers[8].get_weights()[0].reshape((256, 128)).transpose((1, 0)).flatten())))
            f.writelines(list(map(to_string, model.layers[8].get_weights()[1].reshape((128)))))
            f.writelines(list(map(to_string, model.layers[9].get_weights()[0].reshape((128, 10)).transpose((1, 0)).flatten())))
            f.writelines(list(map(to_string, model.layers[9].get_weights()[1].reshape((10)))))

            f.close()
            self.init_weights(open('weights.txt', 'r'))
        except:
            raise Exception('Ошибка во время сохранения весов')

        return 1

    def feedforward(self, values):
        values = np.array(values)

        for layer in self.__layers:
            values = layer.feedforward(values)

        return values

    @property
    def layers(self):
        return self.__layers

def categorical_crossentropy(y_true, y_pred):
    return -(y_true * np.log(y_pred+0.001)).sum(axis=0)

def to_string(s):
    return str(s)+"\n"