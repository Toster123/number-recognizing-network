import os
import keras
import _io
import gc
from typing import Optional
from keras.datasets import mnist
from keras.models import load_model
from sklearn.utils import shuffle
from .layers import *
from .utils import ProgressBridge, to_string, accuracy
from .activations import Softmax

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class SequentalNetwork():
    def __init__(self):
        self.__weights_path = os.path.join(os.getcwd(), 'weights.txt')
        self.__weights_backup_path = os.path.join(os.getcwd(), 'weights_backup.txt')

        if "PARSE_FITTED_WEIGHTS" in os.environ:
            self.parse_trained_weights()

        if not (os.path.exists(self.__weights_path) and (f:=open(self.__weights_path, 'r'))):
            f = None

        self.init_weights(f)

    def parse_trained_weights(self) -> None:
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
                f.writelines(list(map(to_string, model.layers[2].get_weights()[0].reshape((3, 3, 32, 64)).transpose((3, 2, 0, 1)).flatten())))
                f.writelines(list(map(to_string, model.layers[2].get_weights()[1].reshape((64)))))
                f.writelines(list(map(to_string, model.layers[5].get_weights()[0].reshape((5 * 5 * 64, 128)).transpose((1, 0)).flatten())))
                f.writelines(list(map(to_string, model.layers[5].get_weights()[1].reshape((128)))))
                f.writelines(list(map(to_string, model.layers[7].get_weights()[0].reshape((128, 10)).transpose((1, 0)).flatten())))
                f.writelines(list(map(to_string, model.layers[7].get_weights()[1].reshape((10)))))

                f.close()

    def init_weights(self, f: Optional[_io.TextIOWrapper] = None) -> None:
        kernels1 = 0
        shifts1 = 0
        kernels2 = 0
        shifts2 = 0
        weights3 = 0
        shifts3 = 0
        weights4 = 0
        shifts4 = 0

        if f:
            weights = np.array(list(f.readlines())).astype('float32')

            if len(weights) >= 32*3*3 + 32 + 32*64*3*3 + 64 + 128*5*5*64 + 128 + 128*10 + 10:
                kernels1 = weights[:32*3*3].reshape((32, 1, 3, 3))
                shifts1 = weights[32*3*3 : 32*3*3 + 32].reshape((32))

                kernels2 = weights[32*3*3 + 32 : 32*3*3 + 32 + 64*32*3*3].reshape((64, 32, 3, 3))
                shifts2 = weights[32*3*3 + 32 + 64*32*3*3 : 32*3*3 + 32 + 64*32*3*3 + 64].reshape((64))

                weights3 = weights[32*3*3 + 32 + 64*32*3*3 + 64 : 32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64].reshape((128, 5*5*64))
                shifts3 = weights[32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64 : 32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64 + 128].reshape((128))

                weights4 = weights[32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64 + 128 : 32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64 + 128 + 128*10].reshape((10, 128))
                shifts4 = weights[32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64 + 128 + 128*10 : 32*3*3 + 32 + 64*32*3*3 + 64 + 128*5*5*64 + 128 + 128*10 + 10].reshape((10))

        self.__layers = []
        self.__layers.append(Convolution2DLayer((1, 28, 28), (3, 3), 32, kernels1, shifts1))
        self.__layers.append(MaxPooling2DLayer((32, 26, 26), (2, 2)))
        self.__layers.append(Convolution2DLayer((32, 13, 13), (3, 3), 64, kernels2, shifts2))
        self.__layers.append(MaxPooling2DLayer((64, 11, 11), (2, 2)))
        self.__layers.append(FlattenLayer())
        self.__layers.append(DenseLayer(64*5*5, 128, weights3, shifts3))
        self.__layers.append(DropoutLayer(0.5))
        self.__layers.append(DenseLayer(128, 10, weights4, shifts4, activation_func=Softmax()))


    def save_weights(self, f: Optional[_io.TextIOWrapper] = None) -> None:
        if f:
            weights = np.concatenate(tuple([np.concatenate(layer.flatten_weights()) for layer in self.__layers]))
            f.writelines(list(map(to_string, weights)))


    async def fit(self, bridge: ProgressBridge, epochs: int = 10, batch_size: int = 64, dataset_size: float = 0.1, learning_rate: float = 0.01) -> None:

        EPOCHS = epochs
        BATCH_SIZE = batch_size
        DATASET_SIZE = dataset_size
        LEARNING_RATE = learning_rate

        try:
            # with open(self.__weights_backup_path, 'w') as f:
            #     self.save_weights(f)
            #     f.close()
            
            print('Loading dataset')
            (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
            print('Dataset loaded')

            X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
            X_test, Y_test = shuffle(X_test, Y_test, random_state=42)

            X_train = X_train[:int(X_train.shape[0] * DATASET_SIZE)]
            Y_train = Y_train[:int(Y_train.shape[0] * DATASET_SIZE)]
            X_test = X_test[:int(X_test.shape[0] * DATASET_SIZE * 2)]
            Y_test = Y_test[:int(Y_test.shape[0] * DATASET_SIZE * 2)]

            EPOCH_SIZE = X_train.shape[0] // BATCH_SIZE

            X_train = X_train.reshape(X_train.shape[0], 1, 28, 28) / 255.0
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28) / 255.0
            
            Y_train = keras.utils.to_categorical(Y_train, 10)
            Y_test = keras.utils.to_categorical(Y_test, 10)

            for epoch in range(EPOCHS):
                print(f"Ep {epoch+1}")
                
                bridge.add_bar(f"ep_{epoch}", total=EPOCH_SIZE, desc=f"Epoch {epoch+1}/{EPOCHS}")

                X_train, Y_train = shuffle(X_train, Y_train, random_state=42)

                loss_sum = acc_sum = 0

                for batch in range(EPOCH_SIZE):
                    print(f"Batch {batch}")

                    X_batch = X_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
                    Y_batch = Y_train[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]

                    Y_pred = self.forward(X_batch, fitting=True)

                    acc = accuracy(Y_pred, Y_batch)
                    loss, d_logits = Softmax.cross_entropy_loss_and_grads(Y_pred, Y_batch)

                    loss_sum += loss
                    acc_sum += acc

                    d_Z = d_logits
                    for layer in reversed(self.__layers):
                        d_Z = layer.backward(d_Z)

                    for layer in self.__layers:
                        layer.update_weights(LEARNING_RATE)

                    print("Weights updated, collecting")

                    gc.collect()

                    bridge.update(f"ep_{epoch}", n=batch+1, postfix={"loss": round(loss_sum / (batch+1), 5), "acc": round(acc_sum / (batch+1), 5)})
                
                Y_train_pred = self.forward(X_train)
                acc = accuracy(Y_train_pred, Y_train)
                loss, _ = Softmax.cross_entropy_loss_and_grads(Y_train_pred, Y_train)

                Y_test_pred = self.forward(X_test)
                val_acc = accuracy(Y_test_pred, Y_test)
                val_loss, _ = Softmax.cross_entropy_loss_and_grads(Y_test_pred, Y_test)

                print("Metrics got")

                # with open(self.__weights_path, 'w') as f:
                #     self.save_weights(f)
                #     f.close()

                print("Weights file updated")

                bridge.update(f"ep_{epoch}", n=batch+1, postfix={"train_loss": round(loss, 5), "train_acc": round(acc, 5), "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 5)})
                bridge.close_bar(f"ep_{epoch}")

            bridge.mark_finished()
        
        except Exception as e:
            import traceback
            traceback.print_exc()

            bridge.add_bar(f"ep_{epoch+1}", total=1, desc=f"Fitting error: {str(e)}")
            bridge.update(f"ep_{epoch+1}", n=1)
            bridge.close_bar(f"ep_{epoch+1}")
            bridge.mark_finished()

    def forward(self, X: np.ndarray[np.float32 | np.uint8], fitting: bool = False) -> np.ndarray[np.float32]:
        """
        Предсказание цифры
        :param X: матрица изображения с элементами 0-255
        :return: массив вероятностей для каждого класса
        """

        for layer in self.__layers:
            X = layer.forward(X, fitting)

        return X
    
    def __call__(self, X: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return self.forward(X)