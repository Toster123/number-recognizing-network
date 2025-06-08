import numpy as np
from abc import abstractmethod

class Layer():
    def __init__(self, input_size=1):
        self._input_size = input_size
        pass

    @abstractmethod
    def feedforward(self, values):
        pass


class Convolution2DLayer(Layer):
    def __init__(self, input_size=(1, 3, 3), kernel_size=(3, 3), filters_count=1, kernels=0, shifts=0):
        super().__init__(input_size)
        self.__filters_count = filters_count
        self.__kernel_size = kernel_size
        self.__kernels = kernels if kernels else np.random.normal(size=(filters_count, input_size[0], *kernel_size))
        self.__shifts = shifts if shifts else np.random.normal(size=(filters_count))
        self.__output_size = (filters_count, 1 + input_size[1] - kernel_size[0], 1 + input_size[2] - kernel_size[1])

    @property
    def output_size(self):
        return self.__output_size

    @property
    def kernels(self):
        return self.__kernels

    @property
    def shifts(self):
        return self.__shifts

    def feedforward(self, values):
        result = np.empty(self.__output_size)

        for k in range(self.__output_size[0]):
            for i in range(self.__output_size[1]):
                for j in range(self.__output_size[2]):
                    result[k][i][j] = self.activate_relu(np.sum(self.__kernels[k:k+1,:,:,:] * values[:, i:i+self.__kernel_size[0], j:j+self.__kernel_size[1]]) + self.__shifts[k])

        return result

    def activate_relu(self, value):
        return max(0, value)

class MaxPooling2DLayer(Layer):
    # ожидается вход только кратного ядру разрешения
    def __init__(self, input_size=(1, 2, 2), kernel_size=(2, 2)):
        super().__init__(input_size)
        self.__kernel_size = kernel_size
        self.__output_size = (input_size[0], input_size[1] // kernel_size[0], input_size[2] // kernel_size[1])

    def feedforward(self, values):
        result = np.empty(self.__output_size)

        for k in range(self.__output_size[0]):
            for i in range(self.__output_size[1]):
                for j in range(self.__output_size[2]):
                    result[k][i][j] = np.max(values[k, self.__kernel_size[0]*i:self.__kernel_size[0]*i + self.__kernel_size[0], self.__kernel_size[1]*j:self.__kernel_size[1]*j + self.__kernel_size[1]], (0, 1))

        return result


class FlattenLayer(Layer):
    def __init__(self, input_size=(1, 2, 2)):
        super().__init__(input_size)
        self.__output_size = int(np.prod(input_size))

    def feedforward(self, values):
        return values.flatten()


class DenseLayer(Layer):
    def __init__(self, input_size=1, neurons_count=1, weights=0, shifts=0, activation='relu'):
        super().__init__(input_size)
        self.__output_size = neurons_count
        self.__activation = activation
        self.__weights = weights if weights else np.random.normal(size=(neurons_count, input_size))
        self.__shifts = shifts if shifts else np.random.normal(size=(neurons_count))

    def feedforward(self, values):
        result = np.empty(self.__output_size)

        for k in range(self.__output_size):
            result[k] = getattr(self, 'activate_' + str(self.__activation))(sum(values * self.__weights[k]) + self.__shifts[k])


        return result

    def activate_relu(self, value):
        return max(0, value)


class SoftmaxDenseLayer(Layer):
    def __init__(self, input_size=1):
        super().__init__(input_size)
        self.__output_size = input_size

    def feedforward(self, values):
        result = np.exp(values - max(values))
        result /= result.sum(axis=0)

        return result


