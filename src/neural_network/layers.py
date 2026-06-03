import numpy as np
from typing import Optional
from abc import abstractmethod
from scipy.signal import correlate
from .activations import ActivationFunction, ReLU, Softmax


class Layer():
    def __init__(self, input_size: Optional[tuple] = None):
        self._input_size = input_size
        pass

    @abstractmethod
    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        pass

    @abstractmethod
    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass

    @abstractmethod
    def update_weights(self) -> None:
        pass

    @abstractmethod
    def flatten_weights(self) -> np.ndarray[np.float32]:
        pass


class Convolution2DLayer(Layer):
    def __init__(self, input_size: tuple = (1, 3, 3), kernel_size: tuple = (1, 3, 3), filters_count: int = 1, kernels: Optional[np.ndarray[np.float32]] = None, shifts: Optional[np.ndarray[np.float32]] = None):
        super().__init__(input_size)
        self.__kernels = kernels if isinstance(kernels, np.ndarray) else np.random.normal(
            size=(filters_count, input_size[0], *kernel_size))
        self.__shifts = shifts if isinstance(
            shifts, np.ndarray) else np.random.normal(size=(filters_count))
        self.__output_size = (
            filters_count, 1 + input_size[1] - kernel_size[1], 1 + input_size[2] - kernel_size[2])
        
        self.__d_kernels = None
        self.__d_shifts = None

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        result = np.empty((values.shape[0], *self.__output_size))

        for n in range(values.shape[0]):
            for k in range(self.__kernels.shape[0]):
                result[n, k] = correlate(values[n], self.__kernels[k], mode='valid')

        return ReLU.forward(result + self.__shifts[:, np.newaxis, np.newaxis])

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.__kernels.flatten(), self.__shifts.flatten()


class MaxPooling2DLayer(Layer):
    def __init__(self, input_size: tuple = (1, 2, 2), kernel_size: tuple = (2, 2)):
        super().__init__(input_size)
        self.__kernel_size = kernel_size
        self.__output_size = (
            input_size[0], input_size[1] // kernel_size[0], input_size[2] // kernel_size[1])

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        result = np.empty((values.shape[0], *self.__output_size))

        for n in range(values.shape[0]):
            for k in range(self.__output_size[0]):
                for i in range(self.__output_size[1]):
                    for j in range(self.__output_size[2]):
                        result[n, k, i, j] = np.max(values[n, k, self.__kernel_size[0]*i:self.__kernel_size[0]*i +
                            self.__kernel_size[0], self.__kernel_size[1]*j:self.__kernel_size[1]*j + self.__kernel_size[1]])

        return result

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return np.array([]), np.array([])


class FlattenLayer(Layer):
    def __init__(self, input_size: tuple = (1, 2, 2)):
        super().__init__(input_size)

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        return values.transpose((0, 2, 3, 1)).reshape(values.shape[0], -1)

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return np.array([]), np.array([])


class DenseLayer(Layer):
    def __init__(self, input_size: int = 1, neurons_count: int = 1, weights: Optional[np.ndarray[np.float32]] = None, shifts: Optional[np.ndarray[np.float32]] = None, activation_func: type[ActivationFunction] = ReLU):
        super().__init__(input_size)
        self.__activation_func = activation_func
        self.__weights = weights if isinstance(
            weights, np.ndarray) else np.random.normal(size=(neurons_count, input_size))
        self.__shifts = shifts if isinstance(
            shifts, np.ndarray) else np.random.normal(size=(neurons_count))
        
        self.__d_weights = None
        self.__d_shifts = None
        self.__logits = None

    def forward(self, values: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        # todo: if cache_calcs and isinstance(self.__activation_func, Softmax):
        #     self.__logits = values @ self.__weights.T + self.__shifts
        return self.__activation_func.forward(values @ self.__weights.T + self.__shifts)

    def get_logits(self) -> Optional[np.ndarray[np.float32]]:
        return self.__logits

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.__weights.flatten(), self.__shifts.flatten()
