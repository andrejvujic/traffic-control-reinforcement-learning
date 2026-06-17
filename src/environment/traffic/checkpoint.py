
class Checkpoint:
    def __init__(self, position, callback):
        self.position = position
        self.callback = callback

    def __call__(self, vehicle):
        if self.position == vehicle.position():
            self.callback(vehicle)
            return True

        return False
