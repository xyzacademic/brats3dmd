import torch
import argparse
import numpy as np

import os
import torchvision.transforms as transforms

import torch.nn as nn

from dataloader import Brats19Dataset, CropAndPad, Normalize, RandomCrop, RandomFlip
from segmenter import Brats19Segmenter
from lossfunction import DiceLoss, Brats19Loss
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
parser.add_argument('--gpu', default=-1, type=int, help='Using which gpu.')
# parser.add_argument('--threshold', default=0.9, type=float, help='Threshold')
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
config['wd'] = 0.0000
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

data_path = '../source_data/train_data'

current_path = os.getcwd()
if not os.path.exists(config['experiment_name']):
    os.makedirs(config['experiment_name'])

config['save_path'] = os.path.join(config['experiment_name'], os.path.join(args.data, config['save_path']))

########################################################################################
# Pre-processing, dataset, data loader
########################################################################################

data_keyword = ['t1', 'flair', 't1ce', 't2']
target_keyword = ['seg', 'wt', 'tc']


inference_transform = transforms.Compose(
    [
        Normalize(view=config['norm_axis']),
        CropAndPad('Brats19', target='data', channel=len(data_keyword)),

    ]
)
label_transform = transforms.Compose(
    [
        CropAndPad('Brats19', target='label', channel=len(target_keyword)),
    ]
)


test_list = pd.read_csv('statistic/stats.csv')['BraTS_2019_subject_ID'].values

testset = Brats19Dataset(root=data_path, patient_list=test_list, labels=False, nifti=True, data_keyword=data_keyword, target_keyword=target_keyword,
                         data_transform=inference_transform)

test_loader = DataLoader(dataset=testset, batch_size=config['batch_size'], shuffle=False,
                          num_workers=config['batch_size'], pin_memory=True)
########################################################################################
# Network setting
########################################################################################

if args.net == 'mpn':
    from easylab.dl.network.mpn import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=2, norm_axis=config['norm_axis'])
elif args.net == 'mpn_preact':
    # from easylab.dl.network.mpn_preact import MPN
    from mpn_3layer import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=2, norm_axis=config['norm_axis'])
elif args.net == 'mpn_mc':
    from mpn_mc import MPN
    net = MPN(num_channels=len(data_keyword), base_filters=args.basefilter, num_classes=5, norm_axis=config['norm_axis'])

########################################################################################
# Model initialization, loss function setting
########################################################################################
model = Brats19Segmenter(config=config)
model.net_initialize(net)
model.optimizer_initialize()

if config['loss'] == 'diceloss':
    loss = DiceLoss()
if config['loss'] == 'brats19':
    loss = Brats19Loss(sub='diceloss')
if config['loss'] == 'ce':
    loss = nn.CrossEntropyLoss()
model.loss_initialize(loss)

save_path = os.path.join(config['save_path'], "checkpoints")

if config['resume']:
    model.resume(save_path=save_path, filename='ckpt_%d.t7' % config['epoch'])

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

model.optimizer.param_groups[0]['lr'] = config['lr']

model.inference(test_loader=test_loader, prob=False, target_path='prediction', replace_origin='t1', replace_target='pred')



