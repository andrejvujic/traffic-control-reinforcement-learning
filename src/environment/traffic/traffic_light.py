class TrafficLight:
    def __init__(self, index):
        self.index = index
        self.is_enabled = False

    def enable(self):
        self.is_enabled = True

    def disable(self):
        self.is_enabled = False
