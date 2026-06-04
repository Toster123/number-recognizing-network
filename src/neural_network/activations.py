import numpy as np
from abc import abstractmethod


class ActivationFunction():
    def __init__(self):
        pass
    
    @abstractmethod
    def forward(X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        pass

    def __call__(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        return self.forward(X, cache_calcs)

    @abstractmethod
    def backward(d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass
    
    def clear_cache(self):
        pass


class ReLU(ActivationFunction):
    def __init__(self):
        self.__mask = None

    def forward(self, X: np.ndarray[np.float32], cache_calcs: bool = False) -> np.ndarray[np.float32]:
        if cache_calcs:
            self.__mask = X > 0
        return np.maximum(X, 0)

    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return d_Z * self.__mask

    def clear_cache(self):
        self.__mask = None


class Softmax(ActivationFunction):
    def forward(self, logits: np.ndarray[np.float32], _: bool = False) -> np.ndarray[np.float32]:
        exp_z = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_z / exp_z.sum(axis=1)[:, np.newaxis]

        return probs

    def backward(self, d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return d_Z

    @staticmethod
    def cross_entropy_loss_and_grads(probs: np.ndarray[np.float32], Y: np.ndarray[np.float32]) -> tuple[np.float32, np.ndarray[np.float32]]:
        loss = -np.sum(Y * np.log(probs + 1e-12)) / Y.shape[0]
        d_logits = (probs - Y) / Y.shape[0]

        return loss, d_logits