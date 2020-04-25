import sys
sys.path.append('/home/y/yx277/research/easylab')

import torch
import argparse
import numpy as np
from time import time
import os
import torchvision.transforms as transforms
from easylab.dl.network.resnet import ResNet18
from easylab.dl.model import Classifier

from torch.utils.data import DataLoader
import pickle
import sys
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser(description='Train DTS with 2d segmentation')

parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
parser.add_argument('--resume', action='store_true', help='Load pre-trained weights')
parser.add_argument('--batch-size', default=4, type=int, help='Batch size.')
parser.add_argument('--seed', default=4096, type=int, help='Random seed.')
parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate.')
parser.add_argument('--epoch', default=100, type=int, help='The number of epochs')
parser.add_argument('--gpu', default=-1, type=int, nargs='+', help='Using which gpu.')
parser.add_argument('--net', default='ours', type=str, help='which network')
parser.add_argument('--syncbn', action='store_true', help='Using synchronize batchnorm')
parser.add_argument('--cpu', action='store_true', help='Using CPU')
parser.add_argument('--distributed', action='store_true', help='Whether using distributed training')
parser.add_argument('--comment', default='', type=str, help='comment')
args = parser.parse_args()


########################################################################################
# Global Flag
########################################################################################

config = {}

# Config setting
config['resume'] = True if args.resume else False
config['use_cuda'] = False if args.cpu else True
config['fp16'] = args.fp16
config['dtype'] = torch.float16 if config['fp16'] else torch.float32
config['syncbn'] = True if args.syncbn else False
config['gpu'] = args.gpu
config['batch_size'] = args.batch_size
config['seed'] = args.seed
config['lr'] = args.lr
config['wd'] = 0.0001
config['epoch'] = args.epoch
config['lr_decay'] = np.arange(1, config['epoch'])
config['experiment_name'] = args.net
config['cpu'] = True if args.cpu else False
config['distributed'] = True if args.distributed else False
config['save_path'] = "%s" % args.comment


print(sys.argv)
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

########################################################################################
# Data setting
########################################################################################

train_path = '/home/y/yx277/research/ImageDataset/cifar10/train'
test_path = '/home/y/yx277/research/ImageDataset/cifar10/test'
current_path = os.getcwd()
if not os.path.exists(config['experiment_name']):
    os.makedirs(config['experiment_name'])

config['save_path'] = os.path.join(config['experiment_name'], config['save_path'])


train_transform = transforms.Compose(
                [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])

test_transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
                ])


trainset = CIFAR10(root=train_path, train=True, download=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
testset = CIFAR10(root=test_path, train=False, download=True, transform=test_transform)
test_loader = DataLoader(testset, batch_size=4*config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)


net = ResNet18(10, 3)

model = Classifier(config=config)
model.net_initialize(net)
model.optimizer_initialize()

loss = torch.nn.CrossEntropyLoss()
model.loss_initialize(loss)

save_path = os.path.join(config['save_path'], "checkpoints")

if config['resume']:
    model.resume(save_path=save_path, filename='ckpt_%d.t7' % 10)

if config['fp16']:
    opt_level = "O1"
    use_apex = True
else:
    opt_level = "O0"
    use_apex = False
model.manager(use_apex=use_apex, opt_level=opt_level)

start_epoch = 1
methods = ['accuracy']

for epoch in range(start_epoch, start_epoch + config['epoch']):

    model.train(epoch, train_loader, em_list=methods, verbose=True)

    if epoch in [60, 120, 160]:
        model.optimizer.param_groups[0]['lr'] *= 0.1

    model.save(save_path, 'ckpt_%d.t7' % epoch)
    model.test(epoch, test_loader, em_list=methods, verbose=True)





