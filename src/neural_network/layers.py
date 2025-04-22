import numpy as np
from abc import abstractmethod

class Layer():
    def __init__(self):
        pass

    @abstractmethod
    def feedforward(self, values):
        pass

# class DenseLayer():
#     def __init__(self, inputs_count=1, neurons_count=1, weights=0):
#         self.__inputs_count = inputs_count
#         self.__neurons_count = neurons_count
#         if len(weights) == inputs_count


class Convolution2DLayer(Layer):
    def __init__(self, input_size=(3, 3, 1), kernel_size=(3, 3), filters_count=1, shifts=0):
        Layer.__init__(self)
        self.__input_size = input_size
        self.__filters_count = filters_count
        self.__kernel_size = kernel_size
        self.__kernel = np.random.rand(*kernel_size, filters_count)
        self.__shifts = shifts if shifts else np.random.rand(filters_count)
        self.__output_size = (1 + input_size[0] - kernel_size[0], 1 + input_size[1] - kernel_size[1], filters_count)

    @property
    def output_size(self):
        return self.__output_size

    @property
    def kernel(self):
        return self.__kernel

    @property
    def shifts(self):
        return self.__shifts

    def feedforward(self, values):
        result = np.empty(self.__output_size)

        for i in range(self.__output_size[0]):
            for j in range(self.__output_size[1]):
                for k in range(self.__output_size[2]):
                    result[i][j][k] = np.sum(self.__kernel[:,:,k:k+1] * values[i:i+self.__kernel_size[0], j:j+self.__kernel_size[1]]) + self.__shifts[k]

        return result


class MaxPooling2DLayer(Layer):
    def __init__(self, input_size=(2, 2, 1), kernel_size=(2, 2)):
        Layer.__init__(self)
        self.__input_size = input_size
        self.__kernel_size = kernel_size
        self.__output_size = (input_size[0] // kernel_size[0], input_size[1] // kernel_size[1], input_size[2])

    def feedforward(self, values):
        result = np.empty(self.__output_size)

        for i in range(self.__output_size[0]):
            for j in range(self.__output_size[1]):
                result[i][j] = np.max(values[self.__kernel_size[0]*i:self.__kernel_size[0]*i + self.__kernel_size[0], self.__kernel_size[1]*j:self.__kernel_size[1]*j + self.__kernel_size[1]], (0, 1))

        return result


class DenseLayerNeuron():
    def __init__(self, inputs_count=1, weights=0, shift=np.random.normal()):
        self.__inputs_count = inputs_count
        self.__weights = weights if weights and isinstance(weights, tuple) else np.random.rand(inputs_count)
        self.__shift = shift

    def activate(self, inputs):
        return 1/(1+np.exp(-sum([inputs[i] * self.__weights[i] for i in range(0, self.__inputs_count)], self.__shift)))

    def deriv_activate(self, inputs):
        fx = self.activate(inputs)
        return fx * (1 - fx)