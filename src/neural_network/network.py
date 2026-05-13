import os
import asyncio
import keras
from tqdm import tqdm
from keras.api.datasets import mnist
from keras.models import load_model
from .layers import *
from .utils import ProgressBridge

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class SequentalNetwork():
    def __init__(self):
        self.__weights_path = os.path.join(os.getcwd(), 'weights.txt')
        self.__weights_backup_path = os.path.join(os.getcwd(), 'weights_backup.txt')

        if "PTW" in os.environ:
            self.parse_trained_weights()

        if not (os.path.exists(self.__weights_path) and (f:=open(self.__weights_path, 'r'))):
            f=0

        self.init_weights(f)

    def parse_trained_weights(self):
        pth = None
        for file in os.listdir(os.getcwd()):
            if file.endswith(".h5"):
                pth = os.path.join(os.getcwd(), file)
                break
        if pth:
            model = load_model(pth)

            with open(self.__weights_path, 'w') as f:
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
        self.__layers.append(DenseLayer(128, 10, weights7, shifts7, activation_func=activate_softmax))


    def save_weights(self, f=0):
        if f:
            weights = np.concatenate(tuple([np.concatenate(layer.flatten_weights()) for layer in self.__layers]))
            f.writelines(list(map(to_string, weights)))


    async def fit(self, bridge: ProgressBridge, epochs: int = 10, batch_size: int = 64, dataset_size: float = 0.1):

        EPOCHS = epochs
        BATCH_SIZE = batch_size
        DATASET_SIZE = dataset_size

        try:
            # with open(self.__weights_backup_path, 'w') as f:
            #     self.save_weights(f)
            #     f.close()
            
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train[:int(x_train.shape[0] * DATASET_SIZE)]
            y_train = y_train[:int(y_train.shape[0] * DATASET_SIZE)]
            x_test = x_test[:int(x_test.shape[0] * DATASET_SIZE)]
            y_test = y_test[:int(x_test.shape[0] * DATASET_SIZE)]

            EPOCH_SIZE = x_train.shape[0] // BATCH_SIZE

            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

            y_train = keras.utils.to_categorical(y_train, 10)
            y_test = keras.utils.to_categorical(y_test, 10)

            for epoch in range(EPOCHS):
                print(f"ep {epoch}")
                
                bridge.add_bar(f"ep_{epoch}", total=EPOCH_SIZE, desc=f"Epoch {epoch+1}/{EPOCHS}")
                for batch in range(EPOCH_SIZE):
                    #MARK: todo: train
                    await asyncio.sleep(0.1)

                    bridge.update(f"ep_{epoch}")
                
                # with open(self.__weights_path, 'w') as f:
                #     self.save_weights(f)
                #     f.close()
                bridge.close_bar(f"ep_{epoch}")

            bridge.mark_finished()
        
        except Exception as e:
            raise Exception('Fitting error: ' + str(e))


    def feedforward(self, values):
        values = np.array(values)

        for layer in self.__layers:
            values = layer.feedforward(values)

        return values


def categorical_crossentropy(y_true, y_pred):
    return -(y_true * np.log(y_pred+0.001)).sum(axis=0)

def to_string(s):
    return str(s)+"\n"