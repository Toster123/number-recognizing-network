from .screen import Screen
from .neural_network.network import SequentalNetwork


class App():
    def __init__(self):
        network = SequentalNetwork()
        Screen(network)

if __name__ == "__main__":
    app = App()