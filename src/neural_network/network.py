import os
import keras
from .layers import *
from keras.api.datasets import mnist


class SequentalNetwork():
    def __init__(self):
        kernels1=0
        shifts1=0
        kernels2=0
        shifts2=0
        weights3=0
        shifts3=0
        weights4=0
        shifts4=0

        if os.path.exists('weights.txt') and (f:=open('weights.txt', 'r')):
            weights = list(map(float, f.readlines()))

            # kernels1 = np.array()

        self.__layers = []
        self.__layers.append(Convolution2DLayer((1, 28, 28), (3, 3), 32, kernels1, shifts1))
        self.__layers.append(Convolution2DLayer((32, 26, 26), (3, 3), 64, kernels2, shifts2))
        self.__layers.append(MaxPooling2DLayer((64, 24, 24), (2, 2)))#64 12 12
        self.__layers.append(FlattenLayer((64, 12, 12)))
        self.__layers.append(DenseLayer(64*12*12, 128, weights3, shifts3))
        self.__layers.append(DenseLayer(128, 10, weights4, shifts4))
        self.__layers.append(SoftmaxDenseLayer(10))

    def train(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32') / 255

        y_train = keras.utils.to_categorical(y_train, 10)

        learn_rate = 0.001 # 0.005

        for epoch in range(5):#todo:70
            for image, y_true in zip(x_train, y_train):

                y_pred = self.feedforward(image)

                # loss = categorical_crossentropy(y_true, y_pred)
                loss = keras.losses.categorical_crossentropy(y_true, y_pred).numpy()

                print(y_pred, y_true, loss, categorical_crossentropy(y_true, y_pred))
                exit()



            print(f'Epoch {epoch+1} is over')

        return 1

    def backpropogate(self, x_train, y_train):
        pass

    def feedforward(self, values):
        values = np.array(values)

        for layer in self.__layers:
            values = layer.feedforward(values)

        return values

def categorical_crossentropy(y_true, y_pred):
    return -(y_true * np.log(y_pred+0.001)).sum(axis=0)