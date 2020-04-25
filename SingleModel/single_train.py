import torch
import argparse
import numpy as np

import os
import torchvision.transforms as transforms

import torch.nn as nn

from dataloader import Brats19Dataset, CropAndPad, Normalize, RandomCrop, RandomFlip, RandomRotate, RandomIntensity, PartialRandomIntensity
from segmenter import Brats19Segmenter
from lossfunction import DiceLoss, Brats19Loss, Brats19LossV2
from torch.utils.data import DataLoader
import pandas as pd
import sys

parser = argparse.ArgumentParser(description='Train DTS with 2d segmentation')


parser.add_argument('--norm-axis', default='all', type=str, help='Normalization axis')
parser.add_argument('--data', default=0, type=str, help='Data source')
parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--resume', action='store_true', help='Load pre-trained weights')
parser.add_argument('--batch-size', default=4, type=int, help='Batch size.')
parser.add_argument('--seed', default=4096, type=int, help='Random seed.')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate.')
parser.add_argument('--epoch', default=100, type=int, help='The number of epochs')
parser.add_argument('--gpu', default=-1, type=int, nargs='+', help='Using which gpu.')
parser.add_argument('--wd', default=0.0000, type=float, help='Initial weight decay.')
parser.add_argument('--net', default='ours', type=str, help='which network')
parser.add_argument('--loss', default='diceloss', type=str, help='which loss')
parser.add_argument('--schedule', default='s1', type=str, help='Training schedule, s1, s2, s3')
parser.add_argument('--syncbn', action='store_true', help='Using synchronize batchnorm')
parser.add_argument('--basefilter', default=16, type=int, help='Batch size.')
parser.add_argument('--cpu', action='store_true', help='Using CPU')
parser.add_argument('--distributed', action='store_true', help='Whether using distributed training')
parser.add_argument('--flip', action='store_true', help='Whether using random flip')
parser.add_argument('--comment', default='', type=str, help='comment')
parser.add_argument('--target', default='wt', type=str, help='comment')

args = parser.parse_args()


########################################################################################
# Global Flag
########################################################################################

config = {}

# Config setting
config['norm_axis'] = args.norm_axis
config['resume'] = True if args.resume else False
config['use_cuda'] = False if args.cpu else True
config['fp16'] = args.fp16
config['dtype'] = torch.float16 if config['fp16'] else torch.float32
config['syncbn'] = True if args.syncbn else False
config['gpu'] = args.gpu
config['batch_size'] = args.batch_size
config['seed'] = args.seed
config['schedule'] = args.schedule
config['lr'] = args.lr
config['wd'] = args.wd
config['loss'] = args.loss
config['epoch'] = args.epoch
config['lr_decay'] = np.arange(1, config['epoch'])
config['experiment_name'] = args.net
config['cpu'] = True if args.cpu else False
config['distributed'] = True if args.distributed else False
config['save_path'] = "%s" % args.comment
config['flip'] = True if args.flip else False


print(sys.argv)
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

########################################################################################
# Data setting
########################################################################################

data_path = '../source_data/train_data_norm'

current_path = os.getcwd()
if not os.path.exists(config['experiment_name']):
    os.makedirs(config['experiment_name'])

config['save_path'] = os.path.join(config['experiment_name'], os.path.join(args.data, config['save_path']))

########################################################################################
# Pre-processing, dataset, data loader
########################################################################################

data_keyword = ['t1', 'flair', 't1ce', 't2']
target_keyword = ['seg', 'wt', 'tc', 'et', 'ed', 'ncr']

# data_transform = transforms.Compose(
#     [
#         # CropAndPad('Brats19', target='data', channel=len(data_keyword)),
#         # Normalize(view=config['norm_axis']),
#         RandomIntensity(prob=0.5, shift=0.5),
#     ]
# )
data_transform=None
both_transform = transforms.Compose(
    [
        # RandomFlip(view=None, prob=0.5),
        # RandomRotate(prob=0.5),
        RandomCrop(format_='Brats19', shape=(160, 192, 128)),
        PartialRandomIntensity(prob=0.6, shift=0.5),

    ]
)

inference_transform = transforms.Compose(
    [
        # Normalize(view=config['norm_axis']),
        CropAndPad('Brats19', target='data', channel=len(data_keyword)),

    ]
)
label_transform = transforms.Compose(
    [
        CropAndPad('Brats19', target='label', channel=len(target_keyword)),
    ]
)

train_list = pd.read_csv('statistic/stats.csv')['BraTS_2019_subject_ID']
test_list = pd.read_csv('statistic/test_35.csv')['BraTS_2019_subject_ID']

trainset = Brats19Dataset(root=data_path, patient_list=train_list, labels=True, nifti=False, data_keyword=data_keyword,
                          target_keyword=target_keyword,
                         data_transform=data_transform, both_transform=both_transform)

train_loader = DataLoader(dataset=trainset, batch_size=config['batch_size'], shuffle=True,
                          num_workers=config['batch_size'], pin_memory=True)

testset = Brats19Dataset(root=data_path, patient_list=test_list, labels=True, nifti=False, data_keyword=data_keyword,
                         target_keyword=target_keyword,
                         data_transform=inference_transform, target_transform=label_transform)

test_loader = DataLoader(dataset=testset, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['batch_size'], pin_memory=True)
########################################################################################
# Network setting
########################################################################################

if args.net == 'v1':
    from version.v01 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v2':
    from version.v02 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v3':
    from version.v03 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v4':
    from version.v04 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v5':
    from version.v05 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v6':
    from version.v06 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v7':
    from version.v07 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v11':
    from version.v11 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v12':
    from version.v12 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v13':
    from version.v13 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v14':
    from version.v14 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v15':
    from version.v15 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v16':
    from version.v16 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])
elif args.net == 'v17':
    from version.v17 import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])

########################################################################################
# Model initialization, loss function setting
########################################################################################
model = Brats19Segmenter(config=config)
model.net_initialize(net)
save_path = os.path.join(config['save_path'], "checkpoints")
if config['resume']:
    model.resume(save_path=save_path, filename='ckpt_%d.t7' % 150)
model.optimizer_initialize()

if config['loss'] == 'diceloss':
    loss = DiceLoss()
if config['loss'] == 'brats19':
    loss = Brats19Loss(sub='diceloss')
if config['loss'] == 'ce':
    loss = nn.CrossEntropyLoss()
if config['loss'] == 'brats19v2':
    loss = Brats19LossV2(sub='diceloss', beta=3)
model.loss_initialize(loss)






use_apex = False
if config['fp16']:
    opt_level = "O1"
    use_apex = True
else:
    opt_level = "O0"
model.manager(use_apex=use_apex, opt_level=opt_level)

start_epoch = 1
methods = ['Background', 'ET', 'TC', 'WT', 'ED', 'NCR']
# methods = ['dice', 'precision', 'recall']
old_loss = 123
best_loss = 123

# model.optimizer.param_groups[0]['lr'] = config['lr']

for epoch in range(start_epoch, start_epoch + config['epoch']):

    current_loss = model.train(epoch, train_loader, em_list=methods, verbose=True)

    if config['schedule'] == 's1':
        if old_loss - current_loss < 0.000001:
            model.optimizer.param_groups[0]['lr'] *= 0.95
    elif config['schedule'] == 's2':
        if epoch in [30, 60, 90]:
            model.optimizer.param_groups[0]['lr'] *= 0.1

    old_loss = current_loss

    model.save(save_path, 'ckpt_%d.t7' % epoch)
    # test_loss = model.test(epoch, test_loader, em_list=methods, verbose=True)
    # if test_loss < best_loss:
    #     best_loss = test_loss
    #     model.save(save_path, 'ckpt_best.t7')




