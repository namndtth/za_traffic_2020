import matplotlib.pyplot as plt
from lr_finder_stopping_criteria import LRFinderStoppingCriteria


class LRFinder:
    def __init__(self, learner):
        self.results = []
        self.learner = learner

    def find(self, lr_start=1e-6, lr_multiplier=1.1, smoothing=0.3):
        self.learner.iteration(take_step=False)
        if not self.learner.trainer._kv_initialized:
            self.learner.trainer._init_kvstore()

        self.learner.net.save_parameters("lr_finder.params")
        self.learner.trainer.save_states("lr_finder.state")

        lr = lr_start

        stopping_criteria = LRFinderStoppingCriteria(smoothing)
        while True:
            loss = self.learner.iteration(lr)
            self.results.append((lr, loss))
            if stopping_criteria(loss):
                break
            lr = lr * lr_multiplier
        self.learner.net.load_parameters("lr_finder.params", ctx=self.learner.ctx)
        self.learner.trainer.load_states("lr_finder.state")
        return self.results

    def plot(self):
        lrs = [e[0] for e in self.results]
        losses = [e[1] for e in self.results]
        plt.figure(figsize=(6, 8))
        plt.scatter(lrs, losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.xscale('log')
        plt.yscale('log')
        axes = plt.gca()
        axes.set_xlim([lrs[0], lrs[-1]])
        y_lower = min(losses) * 0.8
        y_upper = losses[0] * 4
        axes.set_ylim([y_lower, y_upper])
        plt.show()
