# build-in library
import os
from time import time

# third-party library
import numpy as np
import torch
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torch import nn
import torch.utils.data
import pandas as pd
import apex
from apex import amp
import torch.distributed as dist
import torch.utils.data.distributed
from apex.parallel import DistributedDataParallel as DDP

from dl.log import record_csv
from dl.utils import AverageMeter
from dl.utils import evaluation

"""
Pre-request library
pytorch 1.1.0
apex-0.4 (python-only)
pandas 0.24.1
numpy 1.16.2

"""


class Model(object):
    """
    This model class is used to handle deep neural network training.
    apex will help fp16 training even fp32 training with dynamic loss.
    All parameters which are necessary will be passed by config, a dictionary.
    """

    def __init__(self, config=None):
        if not isinstance(config, dict):
            raise TypeError("config should be a dictionary. "
                            "Got {}".format(type(config)))
        self.config = config
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.use_apex = False
        self.amp = None
        # assert isinstance(self.net, nn.Module)
        # self.net_initialize()
        # self.optimizer_initialize()

    def net_initialize(self, net):
        self.net = net
        config = self.config
        if config['use_cuda']:
            print('Start move net to GPU')
            if 'seed' in config:
                print("Random seed: %d" % config['seed'])
                torch.cuda.manual_seed_all(config['seed'])
            # Set benchmark as True if input's shape
            # and batch size is fixed.
            cudnn.benchmark = True
            if 'deterministic' in config:
                cudnn.deterministic = config['deterministic']
            if config['gpu'][0] == -1:
                torch.cuda.set_device(0)
            else:
                torch.cuda.set_device(config['gpu'][0])
            self.net.cuda()
        print('Network initialized successfully.')

    def optimizer_initialize(self, optimizer=None, params_list=None):
        config = self.config
        params = self.net.parameters() if params_list is None else params_list
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(
                params,
                lr=config['lr'],
                momentum=0.9,
                weight_decay=config['wd'],
                nesterov=True
            )

    def resume(self, save_path, filename):
        if not os.path.exists(save_path):
            raise FileNotFoundError("{} is not found ".format(save_path))
        path = os.path.join(save_path, filename)
        checkpoint = torch.load(path)
        print('Model load {} successfully.'.format(path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['net'])
        else:
            self.net.load_state_dict(checkpoint['net'])
        # if 'optimizer' in checkpoint:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.use_apex and 'amp' in checkpoint:
            amp.load_state_dict(checkpoint['amp'])

    def save(self, save_path, filename):
        if isinstance(self.net, nn.DataParallel):
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        state = {
            'net': state_dict,
            'optimizer': self.optimizer.state_dict()
        }
        if self.use_apex:
            state['amp'] = amp.state_dict()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, filename)
        print('Saving to {}'.format(path))
        torch.save(state, path)

    def loss_initialize(self, loss):
        config = self.config
        self.criterion = loss
        if isinstance(loss, nn.Module):
            if config['use_cuda'] is True:
                self.criterion.cuda()

    def manager(self, use_apex=False, opt_level="O0", sync_path=None):
        config = self.config
        if use_apex:
            if not self.net or not self.optimizer:
                raise AssertionError("Network and optimizer should be assigned")
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
                                                      opt_level=opt_level)
            self.use_apex = True
        if config['gpu'][0] == -1:
            self.net = nn.DataParallel(module=self.net)
        elif len(config['gpu']) > 1:
            self.net = nn.DataParallel(module=self.net, device_ids=config['gpu'])

        elif config['distributed']:
            if config['use_cuda']:
                self.net = torch.nn.parallel.DistributedDataParallel(self.net,
        device_ids=[config['local_rank']], output_device=config['local_rank'])
            else:
                self.net = torch.nn.parallel.DistributedDataParallelCPU(
                    self.net)
            if not os.path.exists(sync_path):
                os.makedirs(sync_path)
            sync_path = os.path.join(sync_path, 'dist.sync')
            print("Initialize Process Group...")
            print('Backend: %s' % config['backend'])
            print('CUDA' if config['use_cuda'] else 'CPU')
            dist.init_process_group(backend=config['backend'],
                                    init_method='file://%s' % sync_path,
                                    rank=config['global_rank'],
                                    world_size=config['world_size']
                                    )


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt