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


from dl.log import record_csv
from dl.utils import AverageMeter
from dl.utils import evaluation
from dl.model.model import Model
from dl.model.model import reduce_tensor


class Classifier(Model):

    def train(self, epoch=0, train_loader=None, em_list=None, verbose=True):
        config = self.config
        log_columns = ['epoch', 'loss'] + em_list
        record_csv(epoch, log_columns, config['save_path'], "train.csv")
        print("\nEpoch: %d" % epoch)
        self.net.train()
        train_loss = AverageMeter('train_loss')
        eval_dict = {}
        if em_list is not None:
            for method in em_list:
                eval_dict[method] = AverageMeter(method)

        start = time()

        for batch_idx, (data, target) in enumerate(train_loader):
            if config['use_cuda']:
                if isinstance(data, list):
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                    target = target.cuda()
                else:
                    data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            outputs = self.net(data)
            loss = self.criterion(outputs, target)
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if config['distributed']:
                train_loss.update(reduce_tensor(loss.data).item(),
                                  target.size(0))
            else:
                train_loss.update(loss.item(), target.size(0))
            if em_list is not None:
                predict = outputs.max(1)[1]
                for method in em_list:
                    value = evaluation(predict, target, method)
                    if config['distributed']:
                        eval_dict[method].update(reduce_tensor(value).item(),
                                                 target.size(0))
                    else:
                        eval_dict[method].update(value.item(),
                                                 target.size(0))

        df = {}
        df['epoch'] = epoch
        print('%s: %.5f' % (train_loss.name, train_loss.avg))
        df['loss'] = train_loss.avg
        print("This epoch cost %.2f seconds" % (time() - start))
        if em_list is not None:
            for method in em_list:
                print("%s: %.5f" %(method, eval_dict[method].avg))
                df[method] = eval_dict[method].avg
        record_csv(epoch, df, config['save_path'], "train.csv")

        # return train_loss.avg
        return eval_dict[method].avg

    def test(self, epoch=0, test_loader=None, em_list=None, verbose=True):
        config = self.config
        log_columns = ['epoch', 'loss'] + em_list
        record_csv(epoch, log_columns, config['save_path'], "test.csv")
        self.net.eval()
        test_loss = AverageMeter('test_loss')
        eval_dict = {}
        if em_list is not None:
            for method in em_list:
                eval_dict[method] = AverageMeter(method)

        start = time()

        for batch_idx, (data, target) in enumerate(test_loader):
            if config['use_cuda']:
                if isinstance(data, list):
                    for i in range(len(data)):
                        data[i] = data[i].cuda()
                    target = target.cuda()
                else:
                    data, target = data.cuda(), target.cuda()

            outputs = self.net(data)
            loss = self.criterion(outputs, target)
            if config['distributed']:
                test_loss.update(reduce_tensor(loss.data).item(),
                                 target.size(0))
            else:
                test_loss.update(loss.item(), target.size(0))
            if em_list is not None:
                predict = outputs.max(1)[1]
                for method in em_list:
                    value = evaluation(predict, target, method)
                    if config['distributed']:
                        eval_dict[method].update(reduce_tensor(value).item(),
                                                 target.size(0))
                    else:
                        eval_dict[method].update(value.item(),
                                                 target.size(0))
        df = {}
        df['epoch'] = epoch
        print('%s: %.5f' % (test_loss.name, test_loss.avg))
        df['loss'] = test_loss.avg
        print("This epoch cost %.2f seconds" % (time() - start))
        if em_list is not None:
            for method in em_list:
                print("%s: %.5f" % (method, eval_dict[method].avg))
                df[method] = eval_dict[method].avg
        record_csv(epoch, df, config['save_path'], "test.csv")

        # return test_loss.avg
        return eval_dict[method].avg