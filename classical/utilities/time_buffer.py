N = 20

class pipeline():

    def __init__(self, dim=N):
        self._pipeline = []
        self._dim = dim

    def update(self, new):
        self._pipeline.append(new)
        if len(self._pipeline) > self._dim:
            self._pipeline.pop(0)
    def average(self):
        return sum(self._pipeline)/len(self._pipeline)
