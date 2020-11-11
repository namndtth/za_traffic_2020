import gluoncv as gcv
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from mxnet import autograd
from matplotlib import pyplot as plt
import traffic_signs as ts
from gluoncv.data.batchify import Tuple, Pad, Stack
import multiprocessing
from learner import Learner
from lr_finder import LRFinder

mx.random.seed(42)

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    # dataset = mx.gluon.data.vision.CIFAR10(train=True).transform_first(transform)
    items, labels, classes = ts.load_json(
        '/home/namnd/personal-workspace/zalo AI Challenge/za_traffic_2020/traffic_train')

    dataset = ts.TrafficSigns(items, labels)

    ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()
    net = gcv.model_zoo.get_model('yolo3_darknet53_custom', pretrained_base=False)

    width, height = 608, 608

    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in
                                                         range(1)]))  # stack image, all targets generated

    train_loader = gluon.data.DataLoader(
        train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=multiprocessing.cpu_count())

    learner = Learner(net=net, data_loader=data_loader, ctx=ctx)
    lr_finder = LRFinder(learner)
    lr_finder.find(lr_start=1e-6)
    lr_finder.plot()
