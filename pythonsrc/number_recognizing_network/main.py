from pythonsrc.screen.screen import Screen
from pythonsrc.neural_network.network import SequentalNetwork


class App():
    def __init__(self):
        network = SequentalNetwork()
        screen = Screen(network)

if __name__ == "__main__":
    app = App()