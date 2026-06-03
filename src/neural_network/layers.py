import numpy as np
from abc import abstractmethod
from scipy.signal import correlate

class Layer():
    def __init__(self, input_size=None):
        self._input_size = input_size
        pass

    @abstractmethod
    def feedforward(self, values):
        pass

    @abstractmethod
    def flatten_weights(self):
        pass


def activate_relu(values: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return np.maximum(values, 0)

def activate_softmax(values: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    result = np.exp(values - np.max(values))
    result /= result.sum(axis=0)

    return result


class Convolution2DLayer(Layer):
    def __init__(self, input_size=(1, 3, 3), kernel_size=(3, 3), filters_count=1, kernels=0, shifts=0):
        super().__init__(input_size)
        self.__kernels = kernels if isinstance(kernels, np.ndarray) else np.random.normal(size=(filters_count, input_size[0], *kernel_size))
        self.__shifts = shifts if isinstance(shifts, np.ndarray) else np.random.normal(size=(filters_count))
        self.__output_size = (filters_count, 1 + input_size[1] - kernel_size[0], 1 + input_size[2] - kernel_size[1])

    def feedforward(self, values):
        result = np.empty(self.__output_size)
        
        for k in range(self.__kernels.shape[0]):
            result[k] = correlate(values, self.__kernels[k], mode='valid')

        return activate_relu(result + self.__shifts[:, np.newaxis, np.newaxis])
    
    def flatten_weights(self):
        return self.__kernels.flatten(), self.__shifts.flatten()


class MaxPooling2DLayer(Layer):
    def __init__(self, input_size=(1, 2, 2), kernel_size=(2, 2)):
        super().__init__(input_size)
        self.__kernel_size = kernel_size
        self.__output_size = (input_size[0], input_size[1] // kernel_size[0], input_size[2] // kernel_size[1])

    def feedforward(self, values):
        result = np.empty(self.__output_size)

        for k in range(self.__output_size[0]):
            for i in range(self.__output_size[1]):
                for j in range(self.__output_size[2]):
                    result[k][i][j] = np.max(values[k, self.__kernel_size[0]*i:self.__kernel_size[0]*i + self.__kernel_size[0], self.__kernel_size[1]*j:self.__kernel_size[1]*j + self.__kernel_size[1]])

        return result

    def flatten_weights(self):
        return np.array([]), np.array([])

class FlattenLayer(Layer):
    def __init__(self, input_size=(1, 2, 2)):
        super().__init__(input_size)

    def feedforward(self, values):
        return values.transpose((1, 2, 0)).flatten()
    
    def flatten_weights(self):
        return np.array([]), np.array([])


class DenseLayer(Layer):
    def __init__(self, input_size=1, neurons_count=1, weights=0, shifts=0, activation_func=activate_relu):
        super().__init__(input_size)
        self.__activation_func = activation_func
        self.__weights = weights if isinstance(weights, np.ndarray) else np.random.normal(size=(neurons_count, input_size))
        self.__shifts = shifts if isinstance(shifts, np.ndarray) else np.random.normal(size=(neurons_count))

    def feedforward(self, values):
        return self.__activation_func(self.__weights @ values + self.__shifts)
    
    def flatten_weights(self):
        return self.__weights.flatten(), self.__shifts.flatten()
