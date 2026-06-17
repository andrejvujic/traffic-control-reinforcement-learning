
class CarCheckpoint:
    def __init__(self, position, callback):
        self.position = position
        self.callback = callback

    def __call__(self, car):
        if self.position == car.position():
            self.callback(car)
            return True

        return False
