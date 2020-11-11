import gluoncv as gcv
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from mxnet import autograd
from matplotlib import pyplot as plt

mx.random.seed(42)


class Learner:
    def __init__(self, net, data_loader, ctx):
        self.net = net
        self.data_loader = data_loader
        self.ctx = ctx

        self.data_loader_iter = iter(self.data_loader)
        self.net.initialize(mx.init.Xavier(), ctx=self.ctx)
        self.trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})
        self.loss_fn = mx.gluon.loss.SoftmaxCrossEntropyLoss()

    def iteration(self, lr=None, take_step=True):
        if lr and (lr != self.trainer.learning_rate):
            self.trainer.set_learning_rate(lr)

        data, label = next(self.data_loader_iter)
        data = data.as_in_context(self.ctx)
        label = label.as_in_context(self.ctx)

        with mx.autograd.record():
            output = self.net(data)
            loss = self.loss_fn(output, label)
        autograd.backward(loss)

        if take_step:
            self.trainer.step(data.shape[0])

        iteration_loss = mx.nd.mean(loss).asscalar()
        return iteration_loss

    def close(self):
        self.data_loader_iter.shutdown()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])
])

dataset = mx.gluon.data.vision.CIFAR10(train=True).transform_first(transform)


class ContinuousBatchSampler:
    def __init__(self, sampler, batch_size):
        self._sampler = sampler
        self._batch_size = batch_size

    def __iter__(self):
        batch = []
        while True:
            for i in self._sampler:
                batch.append(i)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []


sampler = mx.gluon.data.RandomSampler(len(dataset))
batch_sampler = ContinuousBatchSampler(sampler, batch_size=256)
data_loader = mx.gluon.data.DataLoader(dataset, batch_sampler=batch_sampler)


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


ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
net = mx.gluon.model_zoo.vision.resnet18_v2(classes=10)
learner = Learner(net=net, data_loader=data_loader, ctx=ctx)
lr_finder = LRFinder(learner)
lr_finder.find(lr_start=1e-6)
lr_finder.plot()
