import numpy as np
from typing import Optional
from abc import abstractmethod
from .activations import ActivationFunction, ReLU, Softmax
from .utils import im2col, col2im


class Layer():
    def __init__(self, input_size: Optional[tuple] = None):
        self._input_size = input_size
        pass

    @abstractmethod
    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        pass

    @abstractmethod
    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass

    @abstractmethod
    def update_weights(self, lr: float) -> None:
        pass

    @abstractmethod
    def flatten_weights(self) -> np.ndarray[np.float32]:
        pass


class Convolution2DLayer(Layer):
    def __init__(self, input_size: tuple = (1, 3, 3), kernel_size: tuple = (3, 3), filters_count: int = 1, kernels: Optional[np.ndarray[np.float32]] = None, shifts: Optional[np.ndarray[np.float32]] = None):
        super().__init__(input_size)
        self.__kernels = kernels if kernels is not None else np.random.normal(
            size=(filters_count, input_size[0], *kernel_size))
        self.__shifts = shifts if shifts is not None else np.random.normal(size=(filters_count))
        self.__output_size = (
            filters_count, 1 + input_size[1] - kernel_size[0], 1 + input_size[2] - kernel_size[1])
        
        self.__activation_func = ReLU()

        self.__X_shape = None
        self.__X_col = None

        self.__d_kernels = None
        self.__d_shifts = None

    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        """
        Реализация через im2col для оптимизации (превращения свертки в матричное умножение)
        """

        X_col = im2col(X, self.__kernels.shape[2:])

        if cache_calcs:
            self.__X_shape = X.shape
            self.__X_col = X_col

        kernels_col = self.__kernels.reshape((self.__kernels.shape[0], -1))
        
        # (N*H_out*W_out, K)
        Z_flat = X_col @ kernels_col.T + self.__shifts

        # (N, H_out, W_out, K)
        Z = Z_flat.reshape(X.shape[0], self.__output_size[1], self.__output_size[2], self.__output_size[0])
        
        # (N, K, H_out, W_out)
        Z = Z.transpose((0, 3, 1, 2))

        return self.__activation_func(Z, cache_calcs)
    
    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        d_Z = self.__activation_func.backward(d_Z)
        
        N, K, H_out, W_out = d_Z.shape
        
        # Приводим градиент выхода к (N*L, K)
        d_Z_flat = d_Z.transpose((0, 2, 3, 1)).reshape(N * H_out * W_out, K)
        
        # (K, C*kH*kW)
        kernels_col = self.__kernels.reshape(K, -1)
        
        # (K, C*kH*kW)
        self.__d_kernels = (d_Z_flat.T @ self.__X_col) / N
        self.__d_kernels = self.__d_kernels.reshape(self.__kernels.shape)
        
        # Суммирование по батчу и размеру карты признаков
        # (K)
        self.__d_shifts = np.sum(d_Z_flat, axis=0) / N
        
        # (N*L, C*kH*kW)
        d_X_col = d_Z_flat @ kernels_col

        # Собираем обратно в (N, C, H, W)
        d_X = col2im(d_X_col, self.__X_shape, self.__kernels.shape[2:], self.__output_size)

        return d_X

    def update_weights(self, lr: float) -> None:
        self.__kernels -= lr * self.__d_kernels
        self.__shifts -= lr * self.__d_shifts

        self.__X_shape = None
        self.__X_col = None

        self.__d_kernels = None
        self.__d_shifts = None

        self.__activation_func.clear_cache()

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.__kernels.flatten(), self.__shifts.flatten()


class MaxPooling2DLayer(Layer):
    def __init__(self, input_size: tuple = (1, 2, 2), kernel_size: tuple = (2, 2)):
        super().__init__(input_size)
        self.__kernel_size = kernel_size
        self.__output_size = (
            input_size[0], input_size[1] // kernel_size[0], input_size[2] // kernel_size[1])
        
        self.__X_shape = None
        self.__X_cut_shape = None
        self.__mask = None

    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        """
        Преобразование каждой 2-мерной матрицы в 4-мерную с последующим поиском максимума и сохранением маски
        """
        
        X_cut = X[..., :self.__output_size[1]*self.__kernel_size[0], :self.__output_size[2]*self.__kernel_size[1]]

        X_windows = X_cut.reshape((X.shape[0], self.__output_size[0], self.__output_size[1], self.__kernel_size[0], self.__output_size[2], self.__kernel_size[1]))

        X_windows = X_windows.transpose((0, 1, 2, 4, 3, 5))
        Z = X_windows.max(axis=(-2,-1))

        if cache_calcs:
            self.__X_shape = X.shape
            self.__X_cut_shape = X_cut.shape
            self.__mask = (X_windows == Z[..., np.newaxis, np.newaxis])

        return Z
    
    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        d_X_windows = d_Z[..., np.newaxis, np.newaxis] * self.__mask
        

        d_X_cut = d_X_windows.transpose((0, 1, 2, 4, 3, 5))
        d_X_cut = d_X_cut.reshape(self.__X_cut_shape)

        d_X = np.zeros(self.__X_shape, dtype=d_X_cut.dtype)
        d_X[..., :self.__X_cut_shape[-2], :self.__X_cut_shape[-1]] = d_X_cut

        return d_X

    def update_weights(self, lr: float) -> None:
        self.__X_shape = None
        self.__X_cut_shape = None
        self.__mask = None

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return np.array([]), np.array([])


class FlattenLayer(Layer):
    def __init__(self):
        self.__X_shape = None

    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        if cache_calcs:
            self.__X_shape = X.shape
        return X.transpose((0, 2, 3, 1)).reshape(X.shape[0], -1)

    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return d_Z.reshape(self.__X_shape)

    def update_weights(self, lr: float) -> None:
        self.__X_shape = None

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return np.array([]), np.array([])


class DropoutLayer(Layer):
    def __init__(self, rate: float = 0.5):
        """
        Inverted Dropout
        """
        
        self.__rate = rate

        self.__mask = None

    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        if cache_calcs:
            self.__mask = np.random.uniform(size=X.shape) > self.__rate
            X = X * self.__mask / (1 - self.__rate)
        return X

    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return d_Z * self.__mask / (1 - self.__rate)

    def update_weights(self, lr: float) -> None:
        self.__mask = None

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return np.array([]), np.array([])


class DenseLayer(Layer):
    def __init__(self, input_size: int = 1, neurons_count: int = 1, weights: Optional[np.ndarray[np.float32]] = None, shifts: Optional[np.ndarray[np.float32]] = None, activation_func: ActivationFunction = ReLU()):
        super().__init__(input_size)
        self.__weights = weights if weights is not None else np.random.normal(size=(neurons_count, input_size))
        self.__shifts = shifts if shifts is not None else np.random.normal(size=(neurons_count))
        
        self.__activation_func = activation_func

        self.__X = None

        self.__d_weights = None
        self.__d_shifts = None

    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        if cache_calcs:
            self.__X = X
        return self.__activation_func(X @ self.__weights.T + self.__shifts, cache_calcs)

    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        d_Z = self.__activation_func.backward(d_Z)
        N = d_Z.shape[0]

        self.__d_weights = (d_Z.T @ self.__X) / N
        self.__d_shifts = np.sum(d_Z, axis=0) / N

        dX = d_Z @ self.__weights
        return dX

    def update_weights(self, lr: float) -> None:
        self.__weights -= lr * self.__d_weights
        self.__shifts -= lr * self.__d_shifts

        self.__X = None

        self.__d_weights = None
        self.__d_shifts = None

        self.__activation_func.clear_cache()

    def flatten_weights(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        return self.__weights.flatten(), self.__shifts.flatten()
