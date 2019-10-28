from gluoncv import data, utils
from gluoncv.data import VOCDetection
import argparse
from gluoncv.data.transforms import presets            
from mxnet import nd                       
from gluoncv import model_zoo
import mxnet as mx
from mxnet import gluon
import gluoncv as gcv
import time
from matplotlib import pyplot as plt
import numpy as np
from mxnet import autograd
from gluoncv.utils import download, viz
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader

def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader

parser = argparse.ArgumentParser()
parser.add_argument('--rootpath',help='myData/')  
parser.add_argument('--vocyear', help='e.g. 2000')     
args = parser.parse_args()


classes = ['ok', 'a', 'b']
width, height = 512, 512  # suppose we use 512 as base training size                                                                     

class VOCLike(VOCDetection):
    CLASSES = classes

    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

train_dataset =VOCLike(rootargs.rootpath, splits=[(int(args.vocyear), 'train')])
val_dataset =VOCLike(root='/home/ubuntu/fh/', splits=[(int(args.vocyear), 'val')])

#print('label example:')
#print(train_dataset[0][1])
#print('length of dataset:', len(train_dataset))
                                                                       
net = model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
net.reset_class(classes)
net = model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes,pretrained_base=False, transfer='voc')

train_data = get_dataloader(net, train_dataset, 512, 2, 0)
val_data = get_dataloader(net, val_dataset, 512, 2, 0)

try:
    a = mx.nd.zeros((1,), ctx=mx.gpu(0))
    ctx = [mx.gpu(0)]
except:
    ctx = [mx.cpu()]


net.collect_params().reset_ctx(ctx)
trainer = gluon.Trainer(
                    net.collect_params(), 'sgd',
                    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9},
                    update_on_kvstore=None)


mbox_loss = gcv.loss.SSDMultiBoxLoss()
ce_metric = mx.metric.Loss('CrossEntropy')
smoothl1_metric = mx.metric.Loss('SmoothL1')


for epoch in range(0, 100):
    ce_metric.reset()
    smoothl1_metric.reset()
    tic = time.time()
    btic = time.time()
    net.hybridize(static_alloc=True, static_shape=True)
    for i, batch in enumerate(train_data):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
            autograd.backward(sum_loss)
        # since we have already normalized the loss, we don't want to normalize
        # by batch-size anymore
        trainer.step(1)
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])
        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        if i % 20 == 0:
            print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
        btic = time.time()



net.save_parameters('ssd_512_mobilenet1.0_pikachu.params')
test_url = 'https://raw.githubusercontent.com/zackchase/mxnet-the-straight-dope/master/img/pikachu.jpg'
download(test_url, 'pikachu_test.jpg')
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('ssd_512_mobilenet1.0_pikachu.params')
x, image = gcv.data.transforms.presets.ssd.load_test('pikachu_test.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()




