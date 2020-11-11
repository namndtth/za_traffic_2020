class LRFinderStoppingCriteria:
    def __init__(self, smoothing=0.3, min_iter=20):
        self.smoothing = smoothing
        self.min_iter = min_iter
        self.first_loss = None
        self.running_mean = None
        self.counter = 0

    def __call__(self, loss):
        self.counter += 1
        if self.first_loss is None:
            self.first_loss = loss
        if self.running_mean is None:
            self.running_mean = loss
        else:
            self.running_mean = ((1 - self.smoothing) * loss + (self.smoothing * self.running_mean))
        return (self.running_mean > self.first_loss * 2) and (self.counter >= self.min_iter)
