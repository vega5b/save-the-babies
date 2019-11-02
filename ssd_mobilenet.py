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
import logging
from gluoncv.utils.metrics.accuracy import Accuracy
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

import os

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

def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize(static_alloc=True, static_shape=True)
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))
        
parser = argparse.ArgumentParser()
parser.add_argument('--rootpath',help='/home/ubuntu/fh/')  
parser.add_argument('--vocyear', help='e.g. 2000')     
parser.add_argument('--save_interval', type=int,default=1)     
parser.add_argument('--save_prefix',default='')     

args = parser.parse_args()
args.start_epoch=0
args.epochs=30
args.val_interval=1
eval_metric='VOCMApMetric'
classes = ['ok', 'a', 'b']
width, height = 512, 512  # suppose we use 512 as base training size                                                                     

class VOCLike(VOCDetection):
    CLASSES = classes

    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

train_dataset =VOCLike(args.rootpath, splits=[(int(args.vocyear), 'train')])
val_dataset =VOCLike(root='/home/ubuntu/fh/', splits=[(int(args.vocyear), 'val')])
eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
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

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_file_path = args.save_prefix + '_train.log'
log_dir = os.path.dirname(log_file_path)
if log_dir and not os.path.exists(log_dir):
	os.makedirs(log_dir)
fh = logging.FileHandler(log_file_path)
logger.addHandler(fh)
logger.info(args)
logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
best_map = [0]


for epoch in range(args.start_epoch, args.epochs):
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
        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(epoch, (time.time()-tic), name1, loss1, name2, loss2))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0):
        	map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
        	val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        	logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        	current_map = float(mean_ap[-1])
        else:
        	current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)



test_url = 'https://raw.githubusercontent.com/zackchase/mxnet-the-straight-dope/master/img/pikachu.jpg'
download(test_url, 'pikachu_test.jpg')
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=classes, pretrained_base=False)
net.load_parameters('ssd_512_mobilenet1.0_pikachu.params')
x, image = gcv.data.transforms.presets.ssd.load_test('pikachu_test.jpg', 512)
cid, score, bbox = net(x)
ax = viz.plot_bbox(image, bbox[0], score[0], cid[0], class_names=classes)
plt.show()




