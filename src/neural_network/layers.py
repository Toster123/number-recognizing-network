import numpy as np
from abc import abstractmethod
from scipy.signal import correlate


class Layer():
    def __init__(self, input_size=None):
        self._input_size = input_size
        pass

    @abstractmethod
    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        pass

    @abstractmethod
    def backward(self, dZ: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass

    @abstractmethod
    def update_weights(self) -> None:
        pass

    @abstractmethod
    def flatten_weights(self) -> np.ndarray[np.float32]:
        pass


def activate_relu(values: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    return np.maximum(values, 0)


def activate_softmax(values: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    result = np.exp(values - np.max(values, axis=1, keepdims=True))
    result /= result.sum(axis=1)

    return result


class Convolution2DLayer(Layer):
    def __init__(self, input_size=(1, 3, 3), kernel_size=(1, 3, 3), filters_count=1, kernels=0, shifts=0):
        super().__init__(input_size)
        self.__kernels = kernels if isinstance(kernels, np.ndarray) else np.random.normal(
            size=(filters_count, input_size[0], *kernel_size))
        self.__shifts = shifts if isinstance(
            shifts, np.ndarray) else np.random.normal(size=(filters_count))
        self.__output_size = (
            filters_count, 1 + input_size[1] - kernel_size[1], 1 + input_size[2] - kernel_size[2])

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False):
        result = np.empty((values.shape[0], *self.__output_size))

        for n in range(values.shape[0]):
            for k in range(self.__kernels.shape[0]):
                result[n, k] = correlate(values[n], self.__kernels[k], mode='valid')

        return activate_relu(result + self.__shifts[:, np.newaxis, np.newaxis])

    def flatten_weights(self):
        return self.__kernels.flatten(), self.__shifts.flatten()


class MaxPooling2DLayer(Layer):
    def __init__(self, input_size=(1, 2, 2), kernel_size=(2, 2)):
        super().__init__(input_size)
        self.__kernel_size = kernel_size
        self.__output_size = (
            input_size[0], input_size[1] // kernel_size[0], input_size[2] // kernel_size[1])

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False):
        result = np.empty((values.shape[0], *self.__output_size))

        for n in range(values.shape[0]):
            for k in range(self.__output_size[0]):
                for i in range(self.__output_size[1]):
                    for j in range(self.__output_size[2]):
                        result[n, k, i, j] = np.max(values[n, k, self.__kernel_size[0]*i:self.__kernel_size[0]*i +
                            self.__kernel_size[0], self.__kernel_size[1]*j:self.__kernel_size[1]*j + self.__kernel_size[1]])

        return result

    def flatten_weights(self):
        return np.array([]), np.array([])


class FlattenLayer(Layer):
    def __init__(self, input_size=(1, 2, 2)):
        super().__init__(input_size)

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False):
        return values.transpose((0, 2, 3, 1)).reshape(values.shape[0], -1)

    def flatten_weights(self):
        return np.array([]), np.array([])


class DenseLayer(Layer):
    def __init__(self, input_size=1, neurons_count=1, weights=0, shifts=0, activation_func=activate_relu):
        super().__init__(input_size)
        self.__activation_func = activation_func
        self.__weights = weights if isinstance(
            weights, np.ndarray) else np.random.normal(size=(neurons_count, input_size))
        self.__shifts = shifts if isinstance(
            shifts, np.ndarray) else np.random.normal(size=(neurons_count))

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False):
        return self.__activation_func(values @ self.__weights.T + self.__shifts)

    def flatten_weights(self):
        return self.__weights.flatten(), self.__shifts.flatten()
