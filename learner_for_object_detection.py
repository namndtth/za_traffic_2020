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

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()

        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        # objectness, center_targets, scale_targets, weights, class_targets
        fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in
                         range(1, 6)]
        gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
        sum_losses = []
        obj_losses = []
        center_losses = []
        scale_losses = []
        cls_losses = []
        with autograd.record():
            for ix, x in enumerate(data):
                obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix],
                                                                  *[ft[ix] for ft in fixed_targets])
                sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                obj_losses.append(obj_loss)
                center_losses.append(center_loss)
                scale_losses.append(scale_loss)
                cls_losses.append(cls_loss)

                autograd.backward(sum_losses)
        trainer.step(batch_size)


if take_step:
    self.trainer.step(data.shape[0])

iteration_loss = mx.nd.mean(loss).asscalar()
return iteration_loss


def close(self):
    self.data_loader_iter.shutdown()
