import logging
import time
import mxnet as mx
from mxnet import autograd, gluon

batch_size = 2


class Learner:
    def __init__(self, net, data_loader, ctx):
        self.net = net
        self.data_loader = data_loader
        self.ctx = [ctx]

        self.data_loader_iter = iter(self.data_loader)
        self.net.initialize(mx.init.Xavier(), ctx=self.ctx)
        self.trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

    def iteration(self, it, lr=None, take_step=True):
        if lr and (lr != self.trainer.learning_rate):
            self.trainer.set_learning_rate(lr)

        # Setup logger
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # metrics
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')

        # Setup time
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        self.net.hybridize()

        # split data
        batch = next(self.data_loader_iter)

        data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)

        # objectness, center_targets, scale_targets, weights, class_targets
        fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in range(1, 6)]
        gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=self.ctx, batch_axis=0)

        sum_losses = []
        obj_losses = []
        center_losses = []
        scale_losses = []
        cls_losses = []
        with autograd.record():
            for ix, x in enumerate(data):
                obj_loss, center_loss, scale_loss, cls_loss = \
                    self.net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                obj_losses.append(obj_loss)
                center_losses.append(center_loss)
                scale_losses.append(scale_loss)
                cls_losses.append(cls_loss)
            autograd.backward(sum_losses)

        obj_metrics.update(0, obj_losses)
        center_metrics.update(0, center_losses)
        scale_metrics.update(0, scale_losses)
        cls_metrics.update(0, cls_losses)

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info(
            '[Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, '
            '{}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'
                .format(it, self.trainer.learning_rate, 4 / (time.time() - btic),
                        name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        if take_step:
            self.trainer.step(batch_size)

        iteration_loss = mx.nd.mean(sum_losses[-1]).asscalar()
        return iteration_loss

    def close(self):
        self.data_loader_iter.shutdown()
