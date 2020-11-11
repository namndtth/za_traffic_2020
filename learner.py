import mxnet as mx
from mxnet import autograd


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
