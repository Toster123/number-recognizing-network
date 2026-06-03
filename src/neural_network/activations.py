import numpy as np
from abc import abstractmethod


class ActivationFunction():
    def __init__(self):
        pass
    
    @staticmethod
    @abstractmethod
    def forward(values: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass

    @staticmethod
    @abstractmethod
    def backward(d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass


class ReLU(ActivationFunction):
    @staticmethod
    def forward(values: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return np.maximum(values, 0)

    @staticmethod
    def backward(d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        pass


class Softmax(ActivationFunction):
    @staticmethod
    def forward(logits: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        result = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        result /= result.sum(axis=1)

        return result

    @staticmethod
    def backward(d_Z: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        return d_Z

    @staticmethod
    def cross_entropy_loss_and_grads(logits: np.ndarray[np.float32], Y: np.ndarray[np.float32]) -> tuple[np.float32, np.ndarray[np.float32]]:
        result = -(Y * np.log(logits+0.001)).sum(axis=1)
        d_Y = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        d_Y /= d_Y.sum(axis=1)
        d_logits = Y - d_Y
        return result, d_logits